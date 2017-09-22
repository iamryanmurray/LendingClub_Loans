import pandas as pd

loans = pd.read_csv("filtered_loans_2007.csv")

#Check the numbef of null values in each column.
null_counts = loans.isnull().sum()
print(null_counts)


#Remove 'pub_rec_bankruptcies' column and then remove all rows with missing values.

loans = loans.drop('pub_rec_bankruptcies',axis=1)
loans = loans.dropna()
print(loans.dtypes.value_counts())

#Take all object columns and copy them to object_columns_df

object_columns_df = loans.select_dtypes(include = ['object'])
print(object_columns_df.head(1))

#Some columns contain date values that will require significant feature engineering:
#earliest_cr_line, last_credit_pull_d - remove these

loans = loans.drop(['earliest_cr_line','last_credit_pull_d'],axis=1)

#int_rate and revol_util represent numeric values, these can be converted.

loans['int_rate'] = loans['int_rate'].str.rstrip('%').astype('float')
loans['revol_util'] = loans['revol_util'].str.rstrip('%').astype('float')

#Check the number of unique values in home_ownership, verification_status,
#emp_length, term, addr_state

cols = ['home_ownership', 'verification_status', 'emp_length', 'term', 'addr_state']

for col in cols:
    print(object_columns_df[col].value_counts())

#home_ownership, verification_status, emp_length, term, and addr_state
#all contain discrete values.  We can treat emp_length as a numerical column
#because the values are ordered.

#Use mapping dictionary to clean the emp_length column

mapping_dict = {
    "emp_length": {
        "10+ years": 10,
        "9 years": 9,
        "8 years": 8,
        "7 years": 7,
        "6 years": 6,
        "5 years": 5,
        "4 years": 4,
        "3 years": 3,
        "2 years": 2,
        "1 year": 1,
        "< 1 year": 0,
        "n/a": 0
    }
}
loans = loans.replace(mapping_dict)



#Check the unique values in title and purpose

print(object_columns_df['purpose'].value_counts())
print(object_columns_df['title'].value_counts())

#purpose and title contain similar information but we can keep the purpose column
#because it has fewer values and the title column has quality issues since many
#values are repeated with slight variation.

#addr_state contains many discrete values and 49 dummy variable columns would 
#be needed to use it for classification

loans = loans.drop(['addr_state','title',],axis=1)


#Convert home_ownership, verification_status, purpose, and term columns to dummy variables

dummy_cols = ['home_ownership','verification_status','purpose','term']
dummy_df = pd.get_dummies(loans[dummy_cols])
loans = pd.concat([loans,dummy_df],axis=1)
loans = loans.drop(dummy_cols,axis=1)


#Save the engineered dataset as 'cleaned_loans_2007.csv'
loans.to_csv('cleaned_loans_2007.csv', index=False)

#We are now ready to start training machine learning models.

