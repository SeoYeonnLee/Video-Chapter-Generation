import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from memory_cache_utils import MemoryManager

class VideoChapterWindowAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, window_size, dropout=0.1):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {hidden_size} is not a multiple of the number of attention "
                f"heads {num_attention_heads}."
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.window_size = window_size

        self.memory_manager = MemoryManager()
        
        # Linear Layer to transform Query, Key, Value
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        # Output projection
        # self.out_proj = nn.Linear(hidden_size, hidden_size)
        # layer6 - hidden * 2까지
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.LayerNorm(hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        # Dropouts
        self.input_dropout = nn.Dropout(0.1)
        self.attention_dropout = nn.Dropout(0.2)
        # self.output_dropout = nn.Dropout(0.15)

        # Position and progress embedding
        self.position_encoding = nn.Sequential(
            nn.Linear(1, hidden_size//2),  # [normalized_pos, progress]
            nn.LayerNorm(hidden_size//2),
            nn.Tanh(),
            nn.Linear(hidden_size//2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1)
        )

        # Learnable window position bias
        self.window_pos_bias = nn.Parameter(torch.zeros(1, num_attention_heads, 1, window_size))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        scale = 1.0 / math.sqrt(self.attention_head_size)

        # Q, K, V 초기화
        for module in [self.query, self.key, self.value]:
            nn.init.normal_(module.weight, mean=0.0, std=scale)
            # Query는 약간 작게 초기화하여 초기 attention을 더 uniform하게
            if module == self.query:
                module.weight.data *= 0.9
            nn.init.zeros_(module.bias)
        
        # Output projection - 작은 초기값
        # nn.init.normal_(self.out_proj.weight, mean=0.0, std=scale * 0.5)
        # nn.init.zeros_(self.out_proj.bias)
        for layer in self.out_proj:
            if isinstance(layer, nn.Linear):  # Check if the layer is an nn.Linear layer
                nn.init.normal_(layer.weight, mean=0.0, std=scale * 0.5)
                nn.init.zeros_(layer.bias)
        
        # Position encoding 초기화
        for layer in self.position_encoding:
            if isinstance(layer, nn.Linear):
                # Sine-cosine 초기화와 유사한 범위로
                bound = 1 / math.sqrt(layer.weight.shape[1])
                nn.init.uniform_(layer.weight, -bound, bound)
                if layer.bias is not None:
                    nn.init.uniform_(layer.bias, -bound, bound)
            elif isinstance(layer, nn.LayerNorm):
                nn.init.constant_(layer.weight, 1.0)
                nn.init.constant_(layer.bias, 0.0)

        # Window position bias - Transformer 스타일
        nn.init.normal_(self.window_pos_bias, mean=0.0, std=0.02)

    def get_clip_positions(self, clip_indices, total_clips):
        """
        Args:
            clip_indices: window 내 클립들의 전역 인덱스 [window_size]
            total_clips: 비디오의 총 클립 수 (scalar)
        """
        window_size = len(clip_indices)
        middle_idx = window_size // 2

        # 1. Window 내 상대적 위치 (-1 ~ 1)
        local_positions = torch.arange(window_size, device=clip_indices.device) - middle_idx
        local_positions = local_positions.float() / (middle_idx + 1e-6)

        # 2. 전체 비디오에서의 절대적 위치
        # clip_indices = torch.clamp(clip_indices.float(), 0, total_clips-1)
        # global_positions = torch.log(clip_indices + 1) / torch.log(total_clips + 1)

        # 두 위치 정보 결합
        # position_info = torch.stack([local_positions, global_positions], dim=-1)
        # return position_info
        return local_positions.unsqueeze(-1)
        

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_size)

    def forward(self, hidden_states, clip_info):
        """
        Args:
            hidden_states: [batch_size, window_size, hidden_size]
            clip_info: {
                'clip_start_frame': [batch, num_clips],
                'total_frames': [batch],
                'target_clip_idx': [batch],
                'total_num_clips': [batch]
            }
        """
        try:
            batch_size, seq_length, _ = hidden_states.shape
            middle_idx = seq_length // 2  # target clip index
            device = hidden_states.device

            # Calculate position information
            position_info = []
            for i in range(batch_size):
                target_idx = clip_info['target_clip_idx'][i]
                total_clips = clip_info['total_num_clips'][i]
                
                start = target_idx - middle_idx
                window_indices = torch.arange(start, start + seq_length, device=device)
                # window_indices = torch.clamp(window_indices, 0, total_clips - 1)
                
                pos_info = self.get_clip_positions(window_indices, total_clips)
                position_info.append(pos_info)
                
            position_info = torch.stack(position_info, dim=0)  # [batch_size, seq_length, 2]
            
            # Generate position embeddings
            position_embeddings = self.position_encoding(position_info)  # [batch_size, seq_length, hidden_size]

            # Add position embeddings to input
            hidden_states = hidden_states + position_embeddings

            # Generate Q, K, V layer
            target_hidden = hidden_states[:, middle_idx:middle_idx+1, :]  # [batch, 1, hidden]
            query_layer = self.transpose_for_scores(self.query(target_hidden))  # [batch, heads, 1, head_size]
            key_layer = self.transpose_for_scores(self.key(hidden_states))    # [batch, heads, num_clips, head_size]
            value_layer = self.transpose_for_scores(self.value(hidden_states))  # [batch, heads, num_clips, head_size]
            
            # Attention 계산
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # [batch, heads, 1, num_clips]
            attention_scale = math.sqrt(self.attention_head_size) + 1e-6
            attention_scores = attention_scores / attention_scale

            self.window_pos_bias.data = torch.clamp(self.window_pos_bias.data, -10, 10)
            attention_scores = attention_scores + self.window_pos_bias

            attention_scores = torch.clamp(attention_scores, -10, 10)
            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = self.attention_dropout(attention_probs)

            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            context_layer = context_layer.view(batch_size, 1, self.all_head_size)

            # Output projection with dropout
            attention_output = self.out_proj(context_layer) # [batch, 1, hidden]
            # attention_output = self.output_dropout(attention_output)

            # 추론 시에만 중간 결과 정리
            if not self.training:
                intermediate_tensors = [
                    position_info, position_embeddings,
                    query_layer, key_layer, value_layer,
                    attention_scores, attention_probs, context_layer
                ]
                for tensor in intermediate_tensors:
                    del tensor
                torch.cuda.empty_cache()
            
            return attention_output
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                self.memory_manager.cleanup(force=True)
                raise RuntimeError("Memory error in attention computation. Please try with smaller batch size.")
            raise e


class VideoChapterBlock(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, window_size, dropout=0.1):
        super().__init__()
        self.memory_manager = MemoryManager()
        
        self.pre_attention_norm = nn.LayerNorm(hidden_size)
        self.post_attention_norm = nn.LayerNorm(hidden_size)
        self.pre_ffn_norm = nn.LayerNorm(hidden_size)
        self.post_ffn_norm = nn.LayerNorm(hidden_size)

        self.attention = VideoChapterWindowAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            window_size=window_size,
            dropout=dropout
        )
        
        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(0.15)
        )

        self._init_weights()

    def _init_weights(self):
        print("Starting block initialization")
        try:
            # FFN layers 초기화
            for i, module in enumerate(self.ffn):
                print(f"FFN layer {i}: {type(module)}")
                if isinstance(module, nn.Linear):
                    print(f"Linear shape: in={module.in_features}, out={module.out_features}")
                    if module.in_features == module.out_features:
                        nn.init.kaiming_normal_(module.weight, mode='fan_in')
                        module.weight.data *= 0.1
                    else:
                        nn.init.kaiming_normal_(module.weight, mode='fan_out')
                    nn.init.zeros_(module.bias)

            # Layer norms
            for name, module in self.named_modules():
                if isinstance(module, nn.LayerNorm):
                    print(f"Initializing LayerNorm: {name}")
                    nn.init.constant_(module.weight, 1.0)
                    nn.init.constant_(module.bias, 0.0)
                    
            print("Finished block initialization")
        except Exception as e:
            print(f"Error in block init: {str(e)}")
            raise
    
    '''def _init_weights(self):
        # FFN layers 초기화 - Linear 레이어만 초기화
        for module in self.ffn:
            if isinstance(module, nn.Linear):
                if module.in_features == module.out_features:
                    # 크기가 같은 경우 (hidden_size -> hidden_size)
                    nn.init.kaiming_normal_(
                        module.weight,
                        mode='fan_in',
                        nonlinearity='linear'
                    )
                    module.weight.data *= 0.1
                else:
                    # 확장/축소하는 레이어
                    nn.init.kaiming_normal_(
                        module.weight,
                        mode='fan_out',
                        nonlinearity='relu'
                    )
                nn.init.zeros_(module.bias)
        
        # Layer norms 초기화
        for name, module in self.named_modules():
            if isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)'''
    '''def _init_weights(self):
        # First FFN layer - 확장하는 layer
        first_ffn = self.ffn[0]  # hidden_size -> hidden_size * 4
        nn.init.kaiming_normal_(
            first_ffn.weight,
            mode='fan_out',
            nonlinearity='relu'  # GELU용으로도 적합
        )
        nn.init.constant_(first_ffn.bias, 0.01)
        
        # Second FFN layer - 축소하는 layer
        second_ffn = self.ffn[3]  # hidden_size * 4 -> hidden_size
        # 작은 초기값으로 residual connection 효과 유지
        nn.init.kaiming_normal_(
            second_ffn.weight,
            mode='fan_in',
            nonlinearity='linear'
        )
        second_ffn.weight.data *= 0.1
        nn.init.zeros_(second_ffn.bias)
        
        # Layer norms
        for name, module in self.named_modules():
            if isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)'''

    def forward(self, hidden_states, clip_info): # [batch, num_clips, hidden_size]
        try:
            # Attention layer with residual connection
            middle_idx = hidden_states.size(1) // 2
            residual = hidden_states[:, middle_idx:middle_idx+1, :] # [batch, 1, hidden_size]

            # Feed Forward layer with residual connection
            normed_states = self.pre_attention_norm(hidden_states)
            attention_output = self.attention(normed_states, clip_info)  # [batch, 1, hidden_size]
            attention_output = attention_output + residual
            attention_output = self.post_attention_norm(attention_output)

            residual = attention_output
            normed_states = self.pre_ffn_norm(attention_output)
            ffn_output = self.ffn(normed_states)
            output = ffn_output + residual
            output = self.post_ffn_norm(output) # [batch, 1, hidden_size]
            
            if not self.training:
                # 메모리 정리 - 실제 사용한 변수만 삭제
                del normed_states, attention_output, ffn_output
                torch.cuda.empty_cache()

            return output
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                self.memory_manager.cleanup(force=True)
            raise e
    '''def forward(self, hidden_states, clip_info): # [batch, num_clips, hidden_size]
        try:
            # Attention layer with residual connection
            middle_idx = hidden_states.size(1) // 2
            residual = hidden_states[:, middle_idx:middle_idx+1, :] # [batch, 1, hidden_size]

            # Feed Forward layer with residual connection
            normed_states = self.pre_attention_norm(hidden_states)
            attention_output = self.attention(normed_states, clip_info)# [batch, 1, hidden_size]
            attention_output = attention_output + residual
            attention_output = self.post_attention_norm(attention_output)

            residual = attention_output
            normed_states = self.pre_ffn_norm(attention_output)
            ffn_output = self.ffn(normed_states)
            output = ffn_output + residual
            output = self.post_ffn_norm(output) # [batch, 1, hidden_size]
            
            if not self.training:
                del normed_states, attention_output, normed_attention, ffn_output
                torch.cuda.empty_cache()

            return output
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                self.memory_manager.cleanup(force=True)
            raise e'''


class VideoChapterClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.memory_manager = MemoryManager()

        self.window_block = VideoChapterBlock(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            window_size=config.window_size,
            dropout=config.attention_probs_dropout_prob
        )
        
        # Classification head
        # self.classifier = nn.Sequential(
        #     nn.LayerNorm(config.hidden_size),
        #     nn.Dropout(0.1),
        #     nn.Linear(config.hidden_size, config.hidden_size//2),
        #     nn.SiLU(),
        #     nn.LayerNorm(config.hidden_size//2),
        #     nn.Dropout(0.3),
        #     nn.Linear(config.hidden_size//2, 2)
        # )
        # layer 6 - hidden//8까지
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.SiLU(),

            nn.LayerNorm(config.hidden_size),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.SiLU(),

            nn.LayerNorm(config.hidden_size),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.SiLU(),

            nn.LayerNorm(config.hidden_size),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_size, config.hidden_size//2),
            nn.SiLU(),

            nn.LayerNorm(config.hidden_size//2),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_size//2, config.hidden_size//4),
            nn.SiLU(),

            nn.LayerNorm(config.hidden_size//4),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_size//4, config.hidden_size//8),
            nn.SiLU(),

            nn.LayerNorm(config.hidden_size//8),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_size//8, config.hidden_size//16),
            nn.SiLU(),

            nn.Linear(config.hidden_size//16, 2)
        )

        self._init_weights()

    def _init_weights(self):
        print("Starting classifier initialization")
        try:
            # Classifier layers 초기화
            for i, layer in enumerate(self.classifier):
                print(f"Initializing layer {i}: {type(layer)}")
                if isinstance(layer, nn.Linear):
                    print(f"Linear layer shape: in={layer.in_features}, out={layer.out_features}")
                    if layer.out_features == 2:  # 출력 레이어
                        nn.init.xavier_uniform_(layer.weight, gain=1.0)
                        nn.init.zeros_(layer.bias)
                    else:  # 중간 레이어
                        nn.init.kaiming_normal_(
                            layer.weight,
                            mode='fan_out',
                            nonlinearity='relu'
                        )
                        nn.init.constant_(layer.bias, 0.01)
                elif isinstance(layer, nn.LayerNorm):
                    print(f"LayerNorm layer normalized_shape={layer.normalized_shape}")
                    nn.init.constant_(layer.weight, 1.0)
                    nn.init.constant_(layer.bias, 0.0)
            print("Finished classifier initialization")
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise
    '''def _init_weights(self):
        # First classifier layer
        first_layer = self.classifier[1]  # hidden_size -> hidden_size//2
        nn.init.kaiming_normal_(
            first_layer.weight,
            mode='fan_out',
            nonlinearity='relu'
        )
        nn.init.constant_(first_layer.bias, 0.01)
        
        # Output layer - 이진 분류
        output_layer = self.classifier[-1]  # hidden_size//2 -> 2
        # Xavier initialization for logits
        nn.init.xavier_uniform_(
            output_layer.weight,
            gain=1.0  # 로짓이므로 activation 없음
        )
        # Initialize for balanced classes
        nn.init.zeros_(output_layer.bias)
        
        # Layer norms
        for m in self.classifier:
            if isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)'''

    def forward(self, fusion_emb, clip_info):
        device = fusion_emb.device
        if not hasattr(self, '_modules_moved'):
            self.window_block = self.window_block.to(device)
            self.classifier = self.classifier.to(device)
            self._modules_moved = True
        
        try:
            attention_output = self.window_block(fusion_emb, clip_info)
            logits = self.classifier(attention_output.squeeze(1))
            probs = F.softmax(logits, dim=-1)

            if not self.training:
                del attention_output
                torch.cuda.empty_cache()
            
            return logits, probs

        except RuntimeError as e:
            if "out of memory" in str(e):
                self.memory_manager.cleanup(force=True)
            raise e