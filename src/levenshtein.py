import Levenshtein
import csv

def levenshtein_similarity(query):
    database = []
    csv_file = 'data/trucknumber_database.csv'
    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                if row: 
                    database.append(row[0])  
    except FileNotFoundError:
        print(f"오류: '{csv_file}' 파일을 찾을 수 없습니다.")
        return None
    except Exception as e:
        print(f"오류: CSV 파일을 읽는 중 오류 발생: {e}")
        return None
    
    most_similar = None
    max_similarity = 0
    for entry in database:
        # Levenshtein distance 계산
        similarity = Levenshtein.ratio(query, entry)
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar = entry

    return most_similar

if __name__ == "__main__":
    user_query = input("번호판 텍스트를 입력하세요: ")
    best_match = levenshtein_similarity(user_query)
    print(f"입력하신 번호판과 가장 유사한 번호판은: {best_match} 입니다.")