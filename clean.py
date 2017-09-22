import pandas as pd

#Remove the first line because it contains extraneous text.
loans_2007 = pd.read_csv('LoanStats3a.csv', skiprows=1)
#Remove all columns containing more than 50% missing values.
half_count = len(loans_2007) / 2
loans_2007 = loans_2007.dropna(thresh=half_count, axis=1)
#Remove the desc column because it contains a long text explainiation for each loan
loans_2007 = loans_2007.drop(['desc',],axis=1)
#Save the dataset as 'loans_2007.csv'
loans_2007.to_csv('loans_2007.csv', index=False)


#Reimport dataset as loans_2007
loans_2007 = pd.read_csv("loans_2007.csv")

#Remove columns:
col_remove = ['funded_amnt','funded_amnt_inv','grade','sub_grade',
'emp_title','issue_d','zip_code','out_prncp','out_prncp_inv','total_pymnt',
'total_pymnt_inv','total_rec_prncp','total_rec_int','total_rec_late_fee',
'recoveries','collection_recovery_fee','last_pymnt_d','last_pymnt_amnt',]

loans_2007 = loans_2007.drop(col_remove, axis=1)

#funded_amnt - leaks data from the future
#funded_amnt_inv - leaks data from the future
#grade - contains redundant information from interest rate column
#sub_grade - also redundant information
#emp_title - requires too much data processing to be useful
#issue_d - leaks data from the future
#zip_code - redundant with addr_state column since only the first 3 digits of 
#zip code are availble
#out_prncp - leaks data from the future
#out_prncp_inv - leaks data from the future
#total_pymnt - leaks data from the future
#total_pymnt_inv - leaks data from the future
#total_rec_prncp - leaks data from the future
#total_rec_int - leaks data from the future
#total_rec_late_fee - leaks data from the future
#recoveries - leaks data from the future
#collection_recovery_fee - leaks data from the future
#last_pymnt_d - leaks data from the future
#last_pymnt_amnt - leaks data from the future



print(loans_2007.shape)
#Reduced the dataset from 51 columns to 33 columns


#Examine the target colum: loan_status
print(loans_2007['loan_status'].value_counts())

#We are interested in predicting which loans will be fully paid off, and avoiding
#loans that will not be paid.  While 'Default' sounds similar to 'Charged Off', 
#Lending Club considers loans in default to still have a chance of being paid, 
#while loans Charged Off have no chance of being paid.  Therefore, we can use only
#Fully Paid and Charged Off and treat this as a binary classification problem.

#Use a mapping dictionary to map Fully Paid to 1 and Charged Off to 0.

mapping  = {
    'loan_status' : {
        'Fully Paid' : 1,
        'Charged Off' : 0
    }
}

loans_2007 = loans_2007[(loans_2007['loan_status'] == "Fully Paid") |( loans_2007['loan_status'] == "Charged Off")]
loans_2007 = loans_2007.replace(mapping)

#Verify that only values of 0 and 1 are present.
print(loans_2007['loan_status'].value_counts())

#Remove any columns that have only one unique value/

drop_columns = []

for col in loans_2007.columns:
    c = loans_2007[col].dropna()
    if len(c.unique()) == 1:
        drop_columns.append(col)

loans_2007 = loans_2007.drop(drop_columns, axis=1)

print(drop_columns)

#An additional 10 more columns were moved because they contained only one unique value.

#Save dataset as 'filtered_loans_2007.csv'

loans_2007.to_csv('filtered_loans_2007.csv', index=False)
