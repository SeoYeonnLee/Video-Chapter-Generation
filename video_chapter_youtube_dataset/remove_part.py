import os
import glob

def count_and_delete_mp4_part_files(folder_path):
    # .mp4.part로 끝나는 모든 파일의 경로를 가져옵니다
    file_pattern = os.path.join(folder_path, '*.mp4.part')
    files_to_delete = glob.glob(file_pattern)

    # 삭제할 파일의 갯수를 출력합니다
    file_count = len(files_to_delete)
    print(f"삭제할 .mp4.part 파일의 갯수: {file_count}")

    # 사용자에게 삭제 여부를 확인합니다
    user_input = input("이 파일들을 삭제하시겠습니까? (y/n): ")
    
    if user_input.lower() != 'y':
        print("삭제가 취소되었습니다.")
        return

    # 찾은 각 파일을 삭제합니다
    deleted_count = 0
    for file in files_to_delete:
        try:
            os.remove(file)
            print(f"삭제됨: {file}")
            deleted_count += 1
        except Exception as e:
            print(f"파일 삭제 중 오류 발생: {file}")
            print(f"오류 메시지: {str(e)}")

    print(f"총 {deleted_count}개의 .mp4.part 파일이 삭제되었습니다.")

# 스크립트 실행
if __name__ == "__main__":
    vids_folder = "vids"  # vids 폴더의 경로를 여기에 지정하세요
    count_and_delete_mp4_part_files(vids_folder)