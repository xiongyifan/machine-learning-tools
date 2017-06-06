import pandas as pd
from mlt.stats import summary
from mlt.preprocessing import null_value, transform


class Adult():
    data = None

    column_fill_median = []
    column_fill_highest_probability_item = []
    columns_one_hot_encoder = []


    def __init__(self):
        self.read_data('data/adult.csv')

    def classify(self):
        self.column_fill_median = ['fnlwgt', 'education_num', 'age', 'capital_gain', 'capital_loss', 'hours_per_week']
        self.column_fill_highest_probability_item = ['race', 'sex', 'workclass', 'education', 'marital_status', 'occupation',
                                         'relationship', 'native_country']
        self.columns_one_hot_encoder = self.column_fill_highest_probability_item

    def preprocessing(self):
        self.classify()

        self.data = null_value.fill_median(self.data, self.column_fill_median)
        self.data = null_value.fill_highest_probability_item(self.data, self.column_fill_highest_probability_item)

        self.data['income'] = [-1 if x == '<=50K' else 1 for x in self.data['income']]
        X = self.data.drop('income', 1)
        y = self.data.income

        X = transform.one_hot_encoder(X, self.columns_one_hot_encoder)

        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, random_state=1)

        import sklearn.feature_selection

        select = sklearn.feature_selection.SelectKBest(k=20)
        selected_features = select.fit(X_train, y_train)
        indices_selected = selected_features.get_support(indices=True)
        colnames_selected = [X.columns[i] for i in indices_selected]
        #
        X_train_selected = X_train[colnames_selected]
        X_test_selected = X_test[colnames_selected]

        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score


        def find_model_perf(X_train, y_train, X_test, y_test):
            model = LogisticRegression(max_iter=2000)
            model.fit(X_train, y_train)
            y_hat = [x[1] for x in model.predict_proba(X_test)]
            auc = roc_auc_score(y_test, y_hat)

            return auc

        auc_processed = find_model_perf(X_train_selected, y_train, X_test_selected, y_test)
        print(auc_processed)

    def read_data(self, path):
        self.data = pd.read_csv(path, na_values=['#NAME?'])


if __name__ == '__main__':
    adult = Adult()
    summary.summary_data(adult.data)
    adult.preprocessing()
