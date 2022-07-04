from sklearn.linear_model import LogisticRegression

from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector as selector

from imblearn.pipeline import make_pipeline as make_pipeline_with_sampler

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import make_pipeline

# pipe = make_pipeline(StandardScaler(), LogisticRegression())
# impute missing values: numeric values with their mean and boolean values (reported symptoms) with False
num_pipe = make_pipeline(
    StandardScaler(), SimpleImputer(strategy="mean", add_indicator=True)
)

cat_pipe = make_pipeline(
    SimpleImputer(strategy="constant", fill_value=False),
    OneHotEncoder(),
)

preprocessor_linear = make_column_transformer(
    [num_pipe, selector(dtype_include="number")],
    [cat_pipe, selector(dtype_include="object")],
    n_jobs=2,
)

# use polynomial features to enable the model to fit different sets of parameters for different virus variants
# Also use class_weight='balanced' in logistic regression to correct for imbalance between positive and negative test results.

def create_model():
    return make_pipeline_with_sampler(
        preprocessor_linear,
        PolynomialFeatures(interaction_only=True),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )

model = make_pipeline_with_sampler(
    preprocessor_linear,
    PolynomialFeatures(interaction_only=True),
    LogisticRegression(max_iter=1000, class_weight="balanced"),
)
