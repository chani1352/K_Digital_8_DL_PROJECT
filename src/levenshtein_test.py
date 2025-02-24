import Levenshtein
import csv

def levenshtein_similarity_test(str1, str2):
    lev_distance = Levenshtein.distance(str1, str2)
    max_distance = max(len(str1), len(str2))
    similarity = 1 - (lev_distance / max_distance)
    return similarity

if __name__ == "__main__":
    user_query = input("번호판 텍스트를 입력하세요: ")
    best_match = levenshtein_similarity_test(str1, str2)
    print(f"두 문자의 유사도는 : {best_match} 입니다.")