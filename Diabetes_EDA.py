import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/dayallenragunathan/Downloads/diabetes_all_2016.csv')

pd.set_option('display.max_columns', 100)


#print(df.isna().sum())

#print(df.to_string())

#subplot for bw vs bp
plt.subplot(4, 1, 1)
plt.scatter(df['BWAD'], df['BPAD'])
plt.title('Body Weight vs Blood Pressure')
plt.xlabel('Body Weight')
plt.ylabel('Blood Pressure')


plt.subplot(4,1,2)
plt.scatter(df['BWAN'], df['BPAN'])
plt.title('Body Weight(night) vs Blood Pressure(night)')
plt.xlabel('Body Weight')
plt.ylabel('Blood Pressure')




plt.subplot(4,1,3)
plt.scatter(df['BMAD'], df['BPAD'])
plt.title('Body Mass vs Blood Pressure')
plt.xlabel('Body Mass')
plt.ylabel('Blood Pressure')



plt.subplot(4,1,4)
plt.plot(df.corr())


plt.tight_layout()
plt.show()
