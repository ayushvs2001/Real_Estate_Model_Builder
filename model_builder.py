import pandas as pd
import numpy as np
from tkinter import *
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from tkinter import messagebox
from joblib import dump, load

result = []

def generate(name):
    """
    This function used to generate model depend on given dataset
    :param name: name of csv file
    """
    # read data from the read the file you given.
    try:
        housing = pd.read_csv(f"{name}.csv")

        # Give information of dataset
        # housing.info()

        housing["CHAS"].value_counts()

        # It is used to split data for training and testing purpose
        # StratifiedShuffleSplit is used to split data correctly and equally
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(housing, housing['CHAS']):
              strat_train_set = housing.loc[train_index]
              strat_test_set = housing.loc[test_index]

        # drop the Label from column from data set
        housing = strat_train_set.drop("MEDV", axis=1)

        # make column to store labels
        housing_labels = strat_train_set["MEDV"].copy()

        # it is used to automate machine learning workflow
        my_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("std_scalar", StandardScaler())
        ])

        housing_num_tr = my_pipeline.fit_transform(housing)  # housing_num_tr is numpy array

        algorithms = [LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor()]

        models = []

        for algorithm in algorithms:
            model = algorithm
            models.append(model)
            # fit data into algorithm
            model.fit(housing_num_tr, housing_labels)

            # used to get score which desribe the our prediction
            score = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
            rmse_scores = np.sqrt(-score)

            # as the mean is less model is more perfect
            def print_scores(scores):
                  mean = scores.mean()
                  result.append(mean)
                  result.append(scores.std())

            print_scores(rmse_scores)

        model_list = result[::2]
        model_no = model_list.index(min(model_list))

        dump(models[model_no], f"{name}_model.joblib")
    except OSError as e:
            messagebox.showerror("Message", "FILE IS NOT FOUND!")


root = Tk()
root.configure(background='black')
root.title("MODEL BUIDER")

dataset_name_label = Label(root, text="MODEL BUILDER", font="arial", fg="red", bg="yellow")
dataset_name_label.grid(row=0, column=1, padx=(0, 190), pady=(10, 10), ipadx=50, ipady=10)

# label and input field for name && Add button
dataset_name_label = Label(root, text="DATASET NAME :", font="arial")
dataset_name_label.grid(row=1, column=0, padx=(0, 10), pady=(10, 10))

dataset_name = Entry(root, width=25, font="arial")
dataset_name.grid(row=1, column=1, padx=(0,190), pady=(10, 10))


generate_btn = Button(root, text="GENERATE MODEL", font="arial", fg="blue", bg="cyan", padx=28, pady=10,
                             command=lambda: generate(dataset_name.get()))
generate_btn.grid(row=6, column=0, columnspan=2, pady=10)


def efficiency():
    """
    This function show the efficiency of algorithm we used
    """
    try:
        root.withdraw()
        top = Toplevel()
        top.title("ALGORITHM EFFICIENCY")
        top.configure(background='black')

        # label at the top of student database name
        my_label = Label(top, text="ALGORITHM EFFICIENCY", font="arial", fg="blue", bg="cyan")
        my_label.grid(row=0, column=0, padx=(20, 20), pady=(0, 10),ipady=10)

        error_value = Button(top, text="ERROR VALUE:", font="arial", fg="black", bg="white")
        error_value.grid(row=1, column=0, padx=(10, 10), pady=(10, 10))


        def load_data(no):
             error_value["text"] = "ERROR VALUE : {:.2f} +- {:.2f}".format(result[no-1],result[no])


        linear_reg = Button(top, text="Linear Regression", font="arial", fg="red", bg="yellow", padx=50,
                            command=lambda:load_data(1))
        linear_reg.grid(row=2, column=0, padx=(20, 20), pady=(10, 10))

        tree = Button(top, text="Decision Tree Regressor", font="arial", fg="black", bg="green", padx=20,
                            command=lambda:load_data(3))
        tree.grid(row=3, column=0, padx=(20, 20), pady=(10, 10))

        forest = Button(top, text="Random Forest Regressor", font="arial", fg="red", bg="violet", padx=15,
                            command=lambda:load_data(5))
        forest.grid(row=4, column=0, padx=(20, 20), pady=(10, 10))

        def hide_open2():
            root.deiconify()
            top.destroy()

        exit2_btn = Button(top, text="EXIT", font="arial", fg="yellow", bg="red", command=hide_open2)
        exit2_btn.grid(row=6, column=0, columnspan=2, pady=10, padx=(0, 20), ipadx=50)

    except IndexError as e:
        messagebox.showerror("Message", "MODEL IS NOT GENERATED!")

    except:
        messagebox.showerror("Message", "PLEASE TRY AGAIN")




efficiency_btn = Button(root, text="ALGORITHM EFFICIENCY", font="arial", fg="black", bg="violet",  pady=10,
                        command=efficiency)
efficiency_btn.grid(row=7, column=0, columnspan=2, pady=10)

exit_btn = Button(root, text="Exit", command=root.quit, font="arial", fg="white", bg="red", padx=50, pady=10)
exit_btn.grid(row=8, column=0, columnspan=2, pady=10)

root.mainloop()