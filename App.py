import streamlit as st
import pandas as pd

st.title("Web-based Data Mining and Analysis Application Steve Dogas (Π2020186)")

# Upload Data
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "tsv"])
if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.tsv'):
        df = pd.read_csv(uploaded_file, sep='\t')
    df['Label'] = range(1, len(df) + 1)
    st.write("Data Loaded:")
    st.dataframe(df)







from sklearn.feature_selection import SelectKBest, chi2

# Feature Selection Tab
st.sidebar.title("Feature Selection Options")
num_features = st.sidebar.slider("Select Number of Features", min_value=1, max_value=len(df.columns)-1, value=5)

if st.sidebar.button("Run Feature Selection"):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    selector = SelectKBest(chi2, k=num_features)
    X_new = selector.fit_transform(X, y)
    selected_features = selector.get_support(indices=True)
    selected_df = df.iloc[:, selected_features].join(df.iloc[:, -1])
    st.write("Selected Features Dataset:")
    st.dataframe(selected_df)

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import plotly.express as px
import umap
from sklearn.decomposition import PCA

# Visualization Tab
st.sidebar.title("Visualization Options")
viz_option = st.sidebar.selectbox("Choose Visualization", ["2D PCA", "3D PCA", "2D UMAP", "3D UMAP", "EDA Plots"])

if viz_option in ["2D PCA", "3D PCA"]:
    pca = PCA(n_components=3 if "3D" in viz_option else 2)
    components = pca.fit_transform(df.iloc[:, :-1])
    components_df = pd.DataFrame(components, columns=[f'PC{i+1}' for i in range(components.shape[1])])
    components_df['Label'] = df.iloc[:, -1]
    if "3D" in viz_option:
        fig = px.scatter_3d(components_df, x='PC1', y='PC2', z='PC3', color='Label')
    else:
        fig = px.scatter(components_df, x='PC1', y='PC2', color='Label')
    st.plotly_chart(fig)

elif viz_option in ["2D UMAP", "3D UMAP"]:
    reducer = umap.UMAP(n_components=3 if "3D" in viz_option else 2)
    embedding = reducer.fit_transform(df.iloc[:, :-1])
    embedding_df = pd.DataFrame(embedding, columns=[f'UMAP{i+1}' for i in range(embedding.shape[1])])
    embedding_df['Label'] = df.iloc[:, -1]
    if "3D" in viz_option:
        fig = px.scatter_3d(embedding_df, x='UMAP1', y='UMAP2', z='UMAP3', color='Label')
    else:
        fig = px.scatter(embedding_df, x='UMAP1', y='UMAP2', color='Label')
    st.plotly_chart(fig)

elif viz_option == "EDA Plots":
    st.write("Exploratory Data Analysis (EDA) Plots")
    fig1 = px.histogram(df, x=df.columns[-1], title='Distribution of Labels')
    st.plotly_chart(fig1)
    fig2 = px.box(df.iloc[:, :-1], title='Box plot of features')
    st.plotly_chart(fig2)


# Classification Tab
st.sidebar.title("Classification Options")
classifier_name = st.sidebar.selectbox("Choose Classifier", ["K-Nearest Neighbors", "Random Forest"])
parameter_value = st.sidebar.slider("Parameter Value", min_value=1, max_value=20, value=5)

if st.sidebar.button("Run Classification"):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if classifier_name == "K-Nearest Neighbors":
        classifier = KNeighborsClassifier(n_neighbors=parameter_value)
    elif classifier_name == "Random Forest":
        classifier = RandomForestClassifier(n_estimators=parameter_value, random_state=42)
    
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_pred, multi_class='ovr')
    
    st.write(f"Accuracy: {acc}")
    st.write(f"F1 Score: {f1}")
    st.write(f"ROC-AUC: {roc_auc}")



# Info Tab
st.sidebar.title("Πληροφορίες Εφαρμογής")
if st.sidebar.button("Show Info"):
    st.write("""
    ## web-based εφαρμογή για εξόρυξη και ανάλυση δεδομένων με χρήση του Streamlit
    This application allows users to load tabular data, visualize it, and apply machine learning algorithms for feature selection and classification.
    ### Ομάδα:
    -  Στέργιος Δόγκας (Π2020186): Φόρτωση και Προετοιμασία Δεδομένων,Οπτικοποίηση Δεδομένων και Αλγορίθμων Μείωσης Διάστασης
    -  Στέργιος Δόγκας (Π2020186): Υλοποίηση Feature Selection Tab,Υλοποίηση Κατηγοριοποίησης (Classification Tab) 
    -  Στέργιος Δόγκας (Π2020186): Αποτελέσματα και Σύγκριση Αλγορίθμων, Σχεδιασμός και Παρουσίαση Info Tab
    -  Στέργιος Δόγκας (Π2020186): Δοκιμή και Βελτιστοποίηση Εφαρμογής,Ανάπτυξη και Δοκιμή μέσω Docker
    -  Στέργιος Δόγκας (Π2020186): Τεκμηρίωση και Συγγραφή Αναφοράς (LaTeX),Δημιουργία UML Διαγράμματος και Κύκλος Ζωής Έκδοσης Λογισμικού,
        """)




