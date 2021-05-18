# imports
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer


from memoized_property import memoized_property
import mlflow
from mlflow.tracking import MlflowClient
import joblib
MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_NAME = "[FR] [PARIS] [JPRezler] TaxiFare version 1"


class Trainer():
    def __init__(self, X, y, estimator='linear', mlflow = False, distance_type='vectorized'):
        """
            X: pandas DataFrame
            y: pandas Series
            estimator is the type of model, should be in the list : 
            ['linear', 'lasso', 'ridge', 'kneighbors', 'svr', 'random_forest', 'gradient_boosting', 'xboost'], linear by default
            distance_type in list = ['distance', 'vectorized']
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = EXPERIMENT_NAME
        self.mlflow = mlflow
        self.distance_type = distance_type
        self.scorer = make_scorer(compute_rmse)
        self.model_name = estimator
        if estimator == 'lasso':
            self.model = Lasso()
        elif estimator == 'ridge':
            self.model = Ridge()
        elif estimator == 'svr':
            self.model = SVR()
        elif estimator == 'kneighbors':
            self.model = KNeighborsRegressor()
        elif estimator == 'random_forest':
            self.model = RandomForestRegressor()
        elif estimator == 'gradient_boosting':
            self.model = GradientBoostingRegressor()
        elif estimator == 'xboost':
            self.model = XGBRegressor()
        else:
            self.model = LinearRegression()


    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def save_model(self, file='model.sav'):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self.pipeline, file)
        return None
    
    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        if mlflow:
            self.mlflow_log_param('distance', self.distance_type) 
        dist_pipe = Pipeline([
        ('dist_trans', DistanceTransformer(distance_type=self.distance_type)),
        ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        self.pipeline = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', self.model)
        ])
        return self
    

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        if self.mlflow :
            self.mlflow_log_param('model', self.model_name) 
        self.pipeline.fit(self.X, self.y)
        return self
    
    def cross_val(self):
        ''' returns the cross val rmse'''
        cv = cross_val_score(self.pipeline, X, y, cv=5, n_jobs=-1, scoring=self.scorer).mean()
        if self.mlflow :
            self.mlflow_log_metric('cross_val', cv)
        return cv

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        if self.mlflow :
            self.mlflow_log_metric('rmse', rmse)
        return rmse



if __name__ == "__main__":
    # get data
    df = get_data()
    # clean data
    df = clean_data(df)
    # set X and y
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)

    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)

    # train
    model_type = ['linear', 'lasso', 'ridge', 'kneighbors', 'svr', 'random_forest', 'gradient_boosting', 'xboost']
    for model in model_type:
        trainer = Trainer(X_train, y_train, estimator=model, distance_type='distance', mlflow=True)
        trainer.run()
        # evaluate
        print(model)
        print('cross val = ', trainer.cross_val())
        print('rmse val set =', trainer.evaluate(X_val, y_val))
        # save model
        trainer.save_model(file=model+'.sav')
