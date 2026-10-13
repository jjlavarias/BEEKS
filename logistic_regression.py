# Resources
# https://www.youtube.com/watch?v=aL21Y-u0SRs  : yt vid this code is based off of
# https://www.youtube.com/watch?v=HYcXgN9HaTM  : extra resource for comprehension
# https://www.youtube.com/watch?v=4u81xU7BIOc  : Sturms vid

# TO-DO
# - swap out synthetic label for real yes/no column ??
#
# NOTES FROM INSPECTING hive_sensor_data.csv:
#   columns: hive_id, time, temp_sensor1, humid_sensor1, temp_sensor2, humid_sensor2
#   - only 1 hive (A1) and 20 rows in this sample -> too small to draw real
#     conclusions from, this is just to get the pipeline working
#
#   - there is NO yes/no column to encode. sensor1 and sensor2 are basically
#     duplicate readings (corr ~0.99), so there's nothing categorical at all.
#
#   - to keep using logistic regression (which needs a binary target), I made
#     a synthetic label: is humid_sensor1 "high" (1) or "low" (0) relative to
#     its own median. SWAP THIS OUT once you have a real yes/no column
#     (e.g. "swarm_event", "queen_present", etc.) - just point `y` at that
#     column instead and delete the synthetic-label block below.

# imports
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris

# load iris into a datatframe
iris = load_iris()
df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df_iris['target'] = iris.target

# variables & calculations / file reading
df = pd.read_csv('hive_sensor_data.csv')  # was pd.DataFrame(data=...), that just makes a
                                           # DataFrame containing the literal string, not the file

# --- SYNTHETIC TARGET (delete this block once you have a real yes/no column) ---
# turns humid_sensor1 into a 0/1 label: 1 = above-median humidity, 0 = below
df['humid_flag'] = (df['humid_sensor1'] > df['humid_sensor1'].median()).astype(int)
# ---------------------------------------------------------------------------

# if/when you DO have a real yes/no column, this is the pattern to use:
# ans = ['no', 'yes']  # turns nos to 0 and yes to 1
# enc = OrdinalEncoder(categories=[ans])
# df['your_column'] = enc.fit_transform(df[['your_column']])

# training section

# splitting data
# predicting humid_flag (0/1) from temp_sensor1 (this is the "S curve" relationship)
x = df[['temp_sensor1']]
y = df['humid_flag']

# IMPLIMENT K-FOLD CROSS VALIDATION
# NOTE: only 20 rows total, so an 80/20 split leaves ~4 rows to test on.
# fine for checking the code runs, not enough to trust the accuracy number.
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=11)

print(x_train.shape)

# training/ using said data
model = LogisticRegression()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
print("accuracy:", model.score(x_test, y_test))  # acc score

# display graphs and outputs
plt.figure()
plt.scatter(df['temp_sensor1'], df['humid_flag'])  # scatter of predictor vs binary target
plt.xlabel('temp_sensor1')
plt.ylabel('humid_flag (0=low, 1=high)')
plt.title('temp_sensor1 vs humid_flag')

plt.figure()
sns.countplot(x='humid_flag', data=df)  # bar plot of the 0/1 counts
plt.title('Count of humid_flag classes')

plt.show()

print(confusion_matrix(y_test, y_predict))  # rows=actual, cols=predicted; diagonal = correct
print(classification_report(y_test, y_predict))