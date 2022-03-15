
import pandas as pd

data_url = 'https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/ml-basics/grades.csv'
df_students = pd.read_csv(data_url,delimiter=',',header='infer')

df_students

df_students = df_students.dropna(axis=0, how='any')

passes = pd.Series(df_students['Grade'] >= 60)

df_students = pd.concat([df_students, passes.rename("Pass")], axis=1)

df_students

from matplotlib import pyplot as plt

# plt.barh(y=df_students.Name, width=df_students.Grade)
fig = plt.bar(x=df_students.Name, height=df_students.Grade, color='orange')

plt.title('Student Grade')
plt.xlabel('Student')
plt.ylabel('Grade')
plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
plt.xticks(rotation=90)

plt.show()

fig, ax = plt.subplots(1, 2, figsize = (10,4))

ax[0].bar(x=df_students.Name, height=df_students.Grade, color='orange')
ax[0].set_title('Grades')
ax[0].set_xticklabels(df_students.Name, rotation=90)

# Create a pie chart of pass counts on the second axis
pass_counts = df_students['Pass'].value_counts()
ax[1].pie(pass_counts, labels=pass_counts)
ax[1].set_title('Passing Grades')
ax[1].legend(pass_counts.keys().tolist())

# Add a title to the Figure
fig.suptitle('Student Data')

# Show the figure
fig.show()


# Use pandas for plotting
df_students.plot.bar(x='Name', y='StudyHours', color='teal', figsize=(6,4))
plt.show()

var_data = df_students['Grade']

fig = plt.figure(figsize=(10,4))
plt.hist(var_data)

plt.title('Data Distribution')
plt.xlabel('Value')
plt.ylabel('Frequecy')

fig.show()

# stats
var = df_students['Grade']

min_val = var.min()
max_val = var.max()
mean_val = var.mean()
med_val = var.median()
mod_val = var.mode()[0]

print('Minimum:{:.2f}\nMean:{:.2f}\nMedian{:.2f}\nMode:{:.2f}\nMaximum:{:.2f}\n'.format(min_val,
mean_val,
med_val,
mod_val,
max_val))

fig = plt.figure(figsize=(10,4))

plt.hist(var)

plt.axvline(x=min_val, color = 'gray', linestyle='dashed', linewidth=2)
plt.axvline(x=mean_val, color = 'cyan', linestyle='dashed', linewidth = 2)
plt.axvline(x=med_val, color = 'red', linestyle='dashed', linewidth = 2)
plt.axvline(x=mod_val, color = 'yellow', linestyle='dashed', linewidth = 2)
plt.axvline(x=max_val, color = 'gray', linestyle='dashed', linewidth = 2)

plt.title('Data Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')

fig.show()

# boxplot
var

fig = plt.figure(figsize=(10,4))

plt.boxplot(var)

plt.title('Data Distribution')

fig.show()

# combine boxplot and histogram
def show_distribution(var_data):
    
    min_val = var_data.min()
    max_val = var_data.max()
    mean_val = var_data.mean()
    med_val = var_data.median()
    mod_val = var_data.mode()[0]

    print('Minimum:{:.2f}\nMean:{:.2f}\nMedian{:.2f}\nMode:{:.2f}\nMaximum:{:.2f}\n'.format(min_val,
    mean_val,
    med_val,
    mod_val,
    max_val))

    fig, ax = plt.subplots(2, 1, figsize=(10,4))

    ax[0].hist(var_data)
    ax[0].set_ylabel('Frequency')

    ax[0].axvline(x=min_val, color = 'gray', linestyle='dashed', linewidth=2)
    ax[0].axvline(x=mean_val, color = 'cyan', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=med_val, color = 'red', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=mod_val, color = 'yellow', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=max_val, color = 'gray', linestyle='dashed', linewidth = 2)

    ax[1].boxplot(var_data, vert=False)
    ax[1].set_xlabel('Value')

    fig.suptitle('Data Distribution')

    fig.show()

col = df_students['Grade']

show_distribution(col)


# Probability density
def show_density(var_data):
    fig = plt.figure(figsize=(10,4))
    
    var_data.plot.density()

    plt.title('Data Density')

    plt.axvline(x=var_data.mean(), color='cyan', linestyle='dashed', linewidth=2)
    plt.axvline(x=var_data.median(), color='red', linestyle='dashed', linewidth=2)
    plt.axvline(x=var_data.mode()[0], color='yellow', linestyle='dashed', linewidth=2)

    plt.show()

col = df_students['Grade']
show_density(col)