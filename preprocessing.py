from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def preprocess_data(df):
    """
    Cleans and preprocesses the dataset:
    - Drops duplicates
    - Scales 'Amount' feature
    - Drops 'Time' and original 'Amount' columns
    """
    df = df.copy()  # Avoid SettingWithCopyWarning
    df.drop_duplicates(inplace=True)

    scaler = StandardScaler()
    df['Amount_Scaled'] = scaler.fit_transform(df[['Amount']])

    df.drop(['Time', 'Amount'], axis=1, inplace=True)
    return df

def split_data(df):
    """
    Splits the data into training and test sets.
    Uses stratification to preserve class balance in both sets.
    """
    X = df.drop('Class', axis=1)
    y = df['Class']
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

def apply_smote(X_train, y_train):
    """
    Applies SMOTE to balance the training dataset.
    """
    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
    return X_resampled, y_resampled
