### 12181583  김성준 ###
###   Project 2-1   ### 

import pandas as pd

data = pd.read_csv('2019_kbo_for_kaggle_v2.csv')


# 1
for year in range(2015, 2019):
    yearly_data = data[data['year'] == year]
    print(f"{year}년도 안타(H) 상위 10명:")
    print(yearly_data.nlargest(10, 'H')[['batter_name', 'H']])
    print(f"{year}년도 타율(avg) 상위 10명:")
    print(yearly_data.nlargest(10, 'avg')[['batter_name', 'avg']])
    print(f"{year}년도 홈런(HR) 상위 10명:")
    print(yearly_data.nlargest(10, 'HR')[['batter_name', 'HR']])
    print(f"{year}년도 출루율(OBP) 상위 10명:")
    print(yearly_data.nlargest(10, 'OBP')[['batter_name', 'OBP']])
    print("\n")


# 2
data_2018 = data[data['year'] == 2018]
positions = ['포수', '1루수', '2루수', '3루수', '유격수', '좌익수', '중견수', '우익수']  # 포지션 목록
for pos in positions:
    pos_data = data_2018[data_2018['cp'] == pos]
    top_player = pos_data.loc[pos_data['war'].idxmax()]
    print(f"2018년 {pos} 포지션 최고 선수: {top_player['batter_name']} (WAR: {top_player['war']:.3f})")


# 3
correlations = data[['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG', 'salary']].corr()
salary_correlations = correlations['salary'].drop('salary')

print("\n연봉과의 상관관계:")
print(salary_correlations.sort_values(ascending=False).to_string(header=False))

highest_correlation = salary_correlations.idxmax()
print(f"연봉과 상관관계가 가장 높은 변수: {highest_correlation}, {salary_correlations[highest_correlation]}")
