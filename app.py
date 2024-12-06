import os # Used for interacting with the operating system and managing system-level tasks. For example, setting paths or accessing environment variables.
import sys # Used for interacting with the operating system and managing system-level tasks. For example, setting paths or accessing environment variables.
import pandas as pd # A library for data manipulation and analysis. Often used to handle tabular data in ML workflows.
import numpy as np # Provides support for numerical computations, such as arrays and matrix operations.


# Evaluate the performance of a regression model.
from sklearn.metrics import mean_squared_error # Measures the average squared differences between predicted and actual values.
from sklearn.metrics import mean_absolute_error # Measures the average absolute differences between predicted and actual values.
from sklearn.metrics import r2_score # Indicates the proportion of variance explained by the model.
from sklearn.model_selection import train_test_split # Splits data into training and testing subsets.
from sklearn.linear_model import ElasticNet # A regression model combining L1 (Lasso) and L2 (Ridge) penalties.


# URL Parsing
from urllib.parse import urlparse # Breaks down URLs into components like scheme, netloc, path, etc.


# MLflow Setup
import mlflow # A tool to manage the ML lifecycle, including tracking experiments, packaging models, and deploying them.
from mlflow.models.signature import infer_signature # Captures the schema of input data and predictions.
import mlflow.sklearn #  Specialized for tracking and saving models built using sklearn.


import logging # Adds logging functionality for debugging and monitoring.


# This line of code sets an environment variable MLFLOW_TRACKING_URI to a specific URL, which is the location of the MLflow Tracking Server.
os.environ["MLFLOW_TRACKING_URI"] = "http://ec2-54-86-6-109.compute-1.amazonaws.com:5000/"

# initialise logging
logging.basicConfig(level=logging.WARN)
# Purpose: Configures the logging system for the script.

    # Key Argument:
    # level=logging.WARN:
    # Sets the logging level to WARNING.
    # Only messages with severity WARNING, ERROR, or CRITICAL will be logged.
    # This level is typically used to reduce noise in logs by ignoring informational or debug messages unless necessary.
    # Other Logging Levels (in increasing order of severity):

    # DEBUG: Detailed, diagnostic information for debugging.
    # INFO: General operational messages.
    # WARNING: Indicates a potential issue that doesn’t disrupt the program.
    # ERROR: A significant problem that affects functionality.
    # CRITICAL: A severe problem causing the program to fail.

logger = logging.getLogger(__name__)
# Purpose: Creates a logger instance for the current script/module.

    # Key Components:
    # __name__: Represents the name of the current module.
    # If the script is executed as the main program, __name__ will be "__main__".
    # If the script is imported as a module, __name__ will be the module’s name.
    # Why Use getLogger(__name__)?

# Allows fine-grained control over logging.
# You can customize the logging behavior for different modules or scripts independently.


# This function, eval_metrics, calculates and returns three key evaluation metrics for comparing predicted values against actual values in a regression task.
def eval_metrics(actual,pred):
    rmse = np.sqrt(mean_squared_error(actual,pred))
    mae = mean_absolute_error(actual,pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__=="__main__":


    # Data ingestion:
    # reading the dataset: Wine quality dataset
    # This line defines the URL of a CSV file hosted on GitHub, specifically from the MLflow repository. 
    csv_url = ("https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv")


    # This block of code attempts to download and read a CSV file from the specified URL using Pandas. If it fails, it catches the exception and logs an error message.
    try:
        data = pd.read_csv(csv_url, sep = ";")
    except Exception as e:
        logger.exception("Unable to download the data")

    # Split
    train, test = train_test_split(data)
    train_x = train.drop(["quality"], axis = 1)
    test_x = test.drop(["quality"], axis = 1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    # setting hyperparameters for elasticnet
    # len(sys.argv) > 1:

        # Checks if the user has provided at least one additional argument (beyond the script name).
        # If true, the first argument (sys.argv[1]) is read and converted to a float.
        # float(sys.argv[1]):
        # Converts the first user-provided argument into a floating-point number (used as the alpha value).
        # else 0.5:
        # If no argument is provided, the default value of 0.5 is used for alpha.
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    # mlflow
    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x,train_y)

        predicted_qualities = lr.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("elasticNet Model (alpha = {:f}, l1_ratio = {:f}):".format(alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # Remote server setup AWS

        remote_server_uri = "http://ec2-54-86-6-109.compute-1.amazonaws.com:5000/"
        mlflow.set_tracking_uri(remote_server_uri)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                lr, "model", registered_model_name="ElasticNetWineModel"
            )
        else:
            mlflow.sklearn.log_model(lr, "model")

