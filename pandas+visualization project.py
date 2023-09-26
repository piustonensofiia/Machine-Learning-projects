# Відкрити та зчитати файл з даними
import pandas as pd

df = pd.read_csv("Top100-2007.csv")

# Визначити та вивести кількість записів та кількість полів у кожному записі
print(df.shape[0])
df.shape[1]

# Вивести 5 записів, починаючи з К-ого, та 3К+2 останніх записів, де число К визначається днем народження студента та має бути визначено як змінна
# k = int(input("Input the K value(your birthday)"))
# March 5
k = 5
df.iloc[k: ].head()
df.tail(3*k+2)

# Визначити та вивести тип полів кожного запису
df.dtypes

# Визначити записи із пропущеними даними та вивести їх на екран, після чого видалити з датафрейму
print(df.isna().sum())
df = df.dropna()
print(df.isna().sum())

# Очистити текстові поля від зайвих пробілів
i = 0
while i < df.shape[1]:
  if (df.iloc[:, i].dtype == object):
    df.iloc[:, i] = df.iloc[:, i].astype(str).apply(lambda x: x.strip())
  i += 1
df.head()
df.to_csv("to_check_spaces")
# switched task as after using strip() I can't detect nulls

print(df.shape[0])
df.shape[1]

# Визначити поля, які потрібно привести до числового вигляду та зробити це
# I would change Singles Record (Career)(later), Winning Percentage, and Career Earnings
# Switched this task with the previous one. Can not work with null values, unfortunately
# Career Earnings
df["Career Earnings"] = df["Career Earnings"].str.replace("$", "", regex = True).astype(str).astype(int)
df.rename(columns={"Career Earnings": "Career Earnings($)"}, inplace = True)

df.head()

# Winning Percentage
df["Winning Percentage"] = df["Winning Percentage"].str.replace("%", "", regex = True).astype(str).astype(float)/100
df.rename(columns={"Winning Percentage": "Winning probability"}, inplace = True)

df.head()

df.dtypes

# На основі поля Singles Record (Career) ввести нові поля: Total (загальна кількість матчів), Win(victory матч) Lose(non victory матч)
df[["Win", "Lose"]] = df["Singles Record (Career)"].apply(lambda x: pd.Series(str(x).split("-")).astype("int"))
df["Total"] = df.Win+df.Lose
df.head()

# Видалити з датафрейму поля Singles Record (Career) та Link to Wikipedia
df = df.drop(columns=["Singles Record (Career)", "Link to Wikipedia"], axis = 1)
df.head()

# Змінити порядок розташування полів таким чином: Rank, Name, Country, Pts, Total, Win, Lose, Winning Percentage
# Career Earnings($) is lost:( 
df = df[["Rank", "Name", "Country", "Pts", "Total", "Win", "Lose", "Winning probability"]]
df.head()

# Визначити та вивести
# Відсортований за абеткою перелік країн, тенісисти з яких входять у Топ-100
df.sort_values(by = "Country", ascending=True)["Country"].unique()

# Гравця та кількість його очок із найменшою сумою призових
df[["Name", "Pts"]][df.Win == df.Win.min()]

# Гравців та країну, яку вони представляють, кількість виграних матчів у яких дорівнює кількості програних
df[["Name", "Country"]][df.Win == df.Lose]

# Визначити та вивести
# Кількість тенісистів з кожної країни у Топ-100
df.groupby(by = "Country").count()["Name"]

# Середній рейтинг тенісистів з кожної країни
round(df.groupby(by = "Country").mean()["Rank"], 2)

print(df.shape[0])
df.shape[1]

# Побудувати діаграму кількості програних матчів по кожній десятці гравців з Топ-100
import seaborn as sns # based on matplotlib
a = 0
b = 9
top = 1
for h in range(10):
  print(top, "TOP-10", "interval: ", "[", a, ",", b, "]")
  sns.lineplot(data = df.Lose.iloc[a:b], color = "red")
  a += 10
  b += 10
  top += 1
  plt.show()

# Побудувати кругову діаграму сумарної величини призових для кожної країни 
counter = df.Country.drop_duplicates().shape[0]
explode= []
f = 0
while f < counter:
  explode.append(0.2)
  f += 1
import matplotlib.pyplot as plt
df.groupby("Country").sum()["Win"].plot(kind = "pie", figsize = (30, 10), pctdistance = 1.25, labeldistance = 0.7, startangle = 90, shadow = True, explode = explode)
plt.show()

# Побудувати на одному графіку
# Середню кількість очок для кожної країни, середню кількість зіграних матчів тенісистами кожної країни
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 5), sharey = True)
ax1.plot(df.groupby("Country").sum()["Pts"], color = "red")
ax1.set_xlabel("Country")
ax1.set_ylabel("Points(mean)")
ax1.tick_params(labelrotation = 90)

ax2.plot(df.groupby("Country").sum()["Total"], color = "green")
ax1.set_xlabel("Country")
ax2.set_ylabel("Matched(mean)")
ax2.tick_params(labelrotation = 90)
plt.show()