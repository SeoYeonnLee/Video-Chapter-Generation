

def convert_clip_label2cut_point(clip_label_array, clip_frame_num, max_offset):
    enter = False
    begin_sec = 0
    end_sec = 0
    clip_cut_points = []
    for i in range(len(clip_label_array)):
        if clip_label_array[i] == 1 and not enter:
            enter = True
            begin_sec = i*max_offset*2
        
        if clip_label_array[i] == 0 and enter:
            enter = False
            end_sec = (i-1)*max_offset*2 + clip_frame_num
            clip_cut_points.append(round((begin_sec + end_sec - 1) / 2))
            
    return clip_cut_points


def calculate_pr(gt_cut_points, pred_cut_points):
    # recall, recall@3s, recall@5s
    tp, fn = 0, 0
    tp_3, fn_3 = 0, 0
    tp_5, fn_5 = 0, 0
    for i in range(len(gt_cut_points)):
        hit = False
        hit_3 = False
        hit_5 = False
        for j in range(len(pred_cut_points)):
            if gt_cut_points[i] == pred_cut_points[j]:
                hit = True
            if gt_cut_points[i] - 3 <= pred_cut_points[j] <= gt_cut_points[i] + 3:
                hit_3 = True
            if gt_cut_points[i] - 5 <= pred_cut_points[j] <= gt_cut_points[i] + 5:
                hit_5 = True
        
        if hit:
            tp += 1
        else:
            fn += 1
        if hit_3:
            tp_3 += 1
        else:
            fn_3 += 1
        if hit_5:
            tp_5 += 1
        else:
            fn_5 += 1
    
    recall = tp / (tp + fn)
    recall_3 = tp_3 / (tp_3 + fn_3)
    recall_5 = tp_5 / (tp_5 + fn_5)

    # precision, precision@3s
    precision = None
    precision_3 = None
    precision_5 = None
    if len(pred_cut_points) > 0:
        tpp, fp = 0, 0
        tpp_3, fp_3 = 0, 0
        tpp_5, fp_5 = 0, 0
        for i in range(len(pred_cut_points)):
            hit = False
            hit_3 = False
            hit_5 = False
            for j in range(len(gt_cut_points)):
                if pred_cut_points[i] == gt_cut_points[j]:
                    hit = True
                if gt_cut_points[j] - 3 <= pred_cut_points[i] <= gt_cut_points[j] + 3:
                    hit_3 = True
                if gt_cut_points[j] - 5 <= pred_cut_points[i] <= gt_cut_points[j] + 5:
                    hit_5 = True
            
            if hit:
                tpp += 1
            else:
                fp += 1
            if hit_3:
                tpp_3 += 1
            else:
                fp_3 += 1
            if hit_5:
                tpp_5 += 1
            else:
                fp_5 += 1

        precision = tpp / (tpp + fp)
        precision_3 = tpp_3 / (tpp_3 + fp_3)
        precision_5 = tpp_5 / (tpp_5 + fp_5)
    
    return recall, recall_3, recall_5, precision, precision_3, precision_5



if __name__ == "__main__":
    clip_array = [1,0,0,0,1,1,0,0,1,1,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
    result = convert_clip_label2cut_point(clip_array)
    print(result)