

#############
## Imports ##
#############

import pickle
import pandas as pd
import numpy as np
import bnlearn as bn
from test_model import test_model

######################
## Boilerplate Code ##
######################

def load_data():
    """Load train and validation datasets from CSV files."""
    # Implement code to load CSV files into DataFrames
    # Example: train_data = pd.read_csv("train_data.csv")
    train_data = pd.read_csv("train_data.csv")
    val_data = pd.read_csv("validation_data.csv")

    return train_data, val_data

def make_network(df):
    """Define and fit the initial Bayesian Network."""
    # Code to define the DAG, create and fit Bayesian Network, and return the model

    nodes = df.columns
    nodes = sorted(nodes) # just to speed up the process
    nodes.remove("Fare_Category")
    nodes.append("Fare_Category")
    edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            edges.append((nodes[i], nodes[j])) 
    dag = bn.make_DAG(DAG=edges, verbose=0)
    net=bn.parameter_learning.fit(dag,df)
    return net 

def make_pruned_network(df):
    """Define and fit a pruned Bayesian Network."""
    with open(f"base_model.pkl", 'rb') as f:
        model = pickle.load(f)
    model = bn.independence_test(model, df, alpha=0.05, prune=True) 
    return model 

def make_optimized_network(df):
    """Perform structure optimization and fit the optimized Bayesian Network."""
    # with open(f"base_model.pkl", 'rb') as f:
    #     model = pickle.load(f)
    dag = bn.structure_learning.fit(df, methodtype="hc")
    optimized_model = bn.parameter_learning.fit(dag, df)
    return optimized_model


def save_model(fname, model):
    """Save the model to a file using pickle."""
    with open(fname, 'wb') as f:
        pickle.dump(model, f)

def evaluate(model_name, val_df):
    """Load and evaluate the specified model."""
    with open(f"{model_name}.pkl", 'rb') as f:
        model = pickle.load(f)
        # bn.plot(model)
        correct_predictions, total_cases, accuracy = test_model(model, val_df)
        print(f"Total Test Cases: {total_cases}")
        print(f"Total Correct Predictions: {correct_predictions} out of {total_cases}")
        print(f"Model accuracy on filtered test cases: {accuracy:.2f}%")

############
## Driver ##
############

def main():
    # Load data
    train_df, val_df = load_data()

    # # Create and save base model
    base_model = make_network(train_df.copy())
    save_model("base_model.pkl", base_model)

    # Create and save pruned model
    pruned_network = make_pruned_network(train_df.copy())
    save_model("pruned_model.pkl", pruned_network)

    # Create and save optimized model
    optimized_network = make_optimized_network(train_df.copy())
    save_model("optimized_model.pkl", optimized_network)


    evaluate("base_model", val_df)
    evaluate("pruned_model", val_df)
    evaluate("optimized_model", val_df)



    print("[+] Done")

if __name__ == "__main__":
    main()

