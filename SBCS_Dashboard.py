import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import textwrap



st.title('South Bend Code School')
st.markdown('Coding Courses for Adult Learners')

uploaded_file = st.file_uploader('Upload your file here.')

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    uncleaned_data = df.copy()



#### RENAME COLUMNS ####

    # save the initial questions
    questions_list = df.columns.tolist()

    # rename the columns for a cleaner process
    df.rename(columns={
        'What is your date of birth?': 'Date of Birth',
        'What is your gender?': 'Gender',
        'What was your highest level of education when you took this course?': 'Education Level',
        'What was your employment status while taking this course?': 'Employment Status',
        'How much prior coding experience did you have before taking this course? (1 = No experience, 10 = Highly experienced, 5 = May have completed a few small projects )': 'Prior coding level',
        'How satisfied were you with the instructors? (1 = Very Dissatisfied, 10 = Very Satisfied, 5 = Neutral )': 'Instructor Satisfaction', 
        'How difficult did you find the course concepts? (0 = Too easy, 10 = Extremely difficult, 5 = Just right)': 'Course Difficulty',
        'How well did this course prepare you for real-world applications? (1 = Not at all, 10 = Extremely well-prepared, 5 = Some concepts were practical and relevant )': 'Real World Preparation', 
        '''How did your experience level affect your understanding of the course?
        (1-3: "Lacked experience and struggled significantly."
        4-6: "Had some experience but still faced challenges."
        7-10: "Had enough experience to follow along easily.")''': 'Experience Level Effect', 
        'How did you feel about the class speed? (1 = Too slow, 10 = Too fast/rushed, 5 = Just right)': 'Class Speed Rating',
        'Did you experience scheduling conflicts that made it difficult to attend class?': 'Scheduling Conflicts',
        'Did you face any technical barriers while taking this course? (e.g., laptop issues, Zoom link problems, internet connection)': 'Technical Barriers',
        'How motivated were you to complete this course? (1 = Not motivated at all, 10 = Extremely motivated)': 'Motivation Level',
        'To what extent did financial stress affect your ability to attend the course? (1 = No impact, 10 = Made it extremely difficult to continue)': 'Financial Stress Impact',
        'How flexible was your schedule in attending the course? (1 = Not flexible at all, 10 = Very flexible, 5 = Occasional scheduling conflicts)': 'Schedule Flexibility',
        'Would additional tutor or mentor support have been helpful for you?': 'Tutor Support Need',
        'Did you feel that you needed extra time to practice outside of class?': 'Extra Practice Needed',
        'If yes, how much additional practice time did you need per week?': 'Extra Practice Hours',
        'Did you feel that a 90-minute class session allowed you to engage with the material at a comfortable pace and effectively absorb the content?': 'Session Length Satisfaction',
        "If you selected 'No', what was your preferred class length?": 'Preferred Class Length',
        'How many days per week would you have preferred to attend the course?': 'Preferred Class Days',
        'Would you recommend this course to a friend?': 'Recommend To Friend',
        'Fundamentals of Web Development (week 1)': 'WebDev_Fundamentals',
        'Fundamentals of Database Structures (week 2)': 'Database_Fundamentals',
        'Cloud Services (week 3)': 'Cloud_Services',
        'Server-Side Code (week 4)': 'Server_Side_Code',
        'Gen AI': 'GenAI',
        'Personal Interest – I enjoyed learning about this topic.': 'Personal Interest',
        'Career Development – To improve my job prospects, gain new skills, or get a promotion.': 'Career Development',
        'Skill Enhancement – To strengthen my technical skills for future projects or work.': 'Skill Enhancement',
        'Academic Requirement – The content of this course was required for my studies or degree program.': 'Academic Requirement',
        'Networking – To connect with professionals or peers in the field.': 'Networking',
        'Entrepreneurship – To use these skills to start my own business or freelance.': 'Entrepreneurship',
        'Affordable cost': 'Affordable Cost',
        'Interesting course topics': 'Interesting Topics',
        'Job placement support and career opportunities': 'Job Support Opportunities',
        'Hands-on projects and practical learning': 'Hands On Practical Learning',
        'Flexibility (e.g., online/virtual format)': 'Flexible Format',
        'No prior experience required': 'No Experience Required',
        'Other (please specify): __________': 'Other_Specify',
        'Lower cost': 'Lower Cost',
        'More advanced topics and skill-building': 'Advanced Topics Skill Building',
        'Stronger job placement support': 'Stronger Job Support',
        'Certification provided': 'Certification Provided',
        'More flexible scheduling options': 'Flexible Scheduling Options',
        'More hands-on projects and real-world applications': 'More Hands On Real World Experience',
        'What is your first name?': 'First_Name',
        'What is your last name?': 'Last_Name',
        'American Indian or Alaska Native': 'is_ai_an',
        'Asian': 'is_asian',
        'Black or African American': 'is_black',
        'Hispanic or Latino/a/e/x': 'is_hispanic',
        'Middle Eastern or North African': 'is_mena',
        'Native Hawaiian or Other Pacific Islander': 'is_nh_pi',
        'White':'is_white',
        'Prefer not to answer': 'is__prefer_no_answer',
        'Have you enrolled in the “Intro to Careers and Coding” program for adults?':'is_enrolled_co',
        'Where did you first hear about our coding school?\n(Select one category below)': 'referral_source',
        'Which online or social media platform did you use?\n(Select all that apply)': 'social_media',
        'Which educational or career org did you hear from?':'edu_org',
        'Which community or non-profit org referred you?': 'community_org', 
        'Who in your network told you about us?': 'network',
        'How did your experience level affect your understanding of the course?\n(1-3: "Lacked experience and struggled significantly."\n4-6: "Had some experience but still faced challenges."\n7-10: "Had enough experience to follow along easily.")': 'Experience Level Effect',
        'How did you feel about the class speed?\n(1 = Too slow, 10 = Too fast/rushed, 5 = Just right)': 'Class Speed Rating',
        'Did you face any technical barriers while taking this course? (e.g., laptop issues, Zoom link problems, internet connection)\n': 'Technical Barriers',
        'How motivated were you to complete this course? (1 = Not motivated at all, 10 = Extremely motivated)': 'Motivation Level',
        "If you selected 'No', what was your preferred class length?": 'Preferred Class Length',
        'Response Type': 'Response_Type',
        'Start Date (UTC)': 'Start_Date',
        'Stage Date (UTC)': 'Stage_Date',
        'Submit Date (UTC)': 'Submit_Date',
        'Network ID': 'Network_ID',
        'Other.1': 'Other_1',
        'Other.2': 'Other_2',
        'Other.3': 'Other_3',
        'Other.4': 'Other_4',
        'Other.5': 'Other_5',
        'Other.6': 'Other_6',
        'Other.7': 'Other_7',
        'Other.8': 'Other_8',
        'Other.9': 'Other_9',
        'Other.10': 'Other_10'
    }, inplace=True)

    # create a data frame of the initial questions and new column titles
    questions_df = pd.DataFrame({
        'Original_Question': questions_list,
        'New_Column_Title': df.columns
    })




###
### DATA CLEANING ###
###


    # Calculate the age of the students
    df['Age'] = df['Submit_Date'] - df['Date of Birth']
    df['Age'] = df['Age'].dt.days // 365
    df['Age'] = df['Age'].astype(int)

    # save variable types
    df['Gender'] = df['Gender'].astype('category')
    df['Education Level'] = df['Education Level'].astype('category')
    df['Employment Status'] = df['Employment Status'].astype('category')
    df['Prior coding level'] = df['Prior coding level'].astype('int')

    # cleaning row by row
    for row in range(0, len(df)):

         # Get race/ethnicity into one column
        if df.loc[row, 'is_ai_an'] == True:
            df.loc[row, 'Ethnicity/Race'] = 'American Indian or Alaska Native'
        if df.loc[row, 'is_asian'] == True:
            df.loc[row, 'Ethnicity/Race'] = 'Asian'
        if df.loc[row, 'is_black'] == True:
            df.loc[row, 'Ethnicity/Race'] = 'Black or African American'
        if df.loc[row, 'is_hispanic'] == True:
            df.loc[row, 'Ethnicity/Race'] = 'Hispanic or Latino/a/e/x'
        if df.loc[row, 'is_mena'] == True:
            df.loc[row, 'Ethnicity/Race'] = 'Middle Eastern or North African'
        if df.loc[row, 'is_nh_pi'] == True:
            df.loc[row, 'Ethnicity/Race'] = 'Native Hawaiian or Other Pacific Islander'
        if df.loc[row, 'is_white'] == True:
            df.loc[row, 'Ethnicity/Race'] = 'White'
        if df.loc[row, 'is__prefer_no_answer'] == True:
            df.loc[row, 'Ethnicity/Race'] = 'Prefer not to answer'

        # Make the Yes/No response for Sheduling Conflicts into a boolean
        if df.loc[row, 'Scheduling Conflicts'] == 'Yes':
            df.loc[row, 'Scheduling Conflicts'] = True
        elif df.loc[row, 'Scheduling Conflicts'] == 'No':
            df.loc[row, 'Scheduling Conflicts'] = False

        # calculate attendance percentage
        df.loc[row, 'Attendance'] = 0
        if df.loc[row, 'WebDev_Fundamentals'] == True:
            df.loc[row, 'Attendance'] += 1
        if df.loc[row, 'Database_Fundamentals'] == True:
            df.loc[row, 'Attendance'] += 1
        if df.loc[row, 'Cloud_Services'] == True:
            df.loc[row, 'Attendance'] += 1
        if df.loc[row, 'Server_Side_Code'] == True:
            df.loc[row, 'Attendance'] += 1
        df.loc[row, 'Attendance'] = df.loc[row, 'Attendance']/4 *100 # get the percentage
    
    # add Attendance to the question dictionary
    questions_df.loc[len(questions_df)] = ['Calculated attendance percentage for the 4 week course.', 'Attendance']

    # save Ethnicity/Race as a category
    df['Ethnicity/Race'] = df['Ethnicity/Race'].astype('category')
    # add Ethnicity/Race to the question dictionary
    questions_df.loc[len(questions_df)] = ['What is the ethncicity/race you identify with?', 'Ethnicity/Race']

    # get rid of unnecessary columns
    drop_cols = ['Other', 'Other_1', 'Other_2', 'Other_3', 'Other_4', 'Other_5', 'Other_6', 'Other_7', 'Other_8', 'Other_9', 'Other_10', 'Other_Specify', 'Response_Type', 'Start_Date', 'Stage_Date', 'Network_ID', 'Tags']
    df.drop(drop_cols, axis=1, inplace=True)

    # Add a column for age brackets
    for row in range(0, len(df)):
        if df.loc[row, 'Age'] < 25:
            df.loc[row, 'Age Bracket'] = '0-25'
        elif df.loc[row, 'Age'] >= 25 and df.loc[row, 'Age'] < 35:
            df.loc[row, 'Age Bracket'] = '25-35'
        elif df.loc[row, 'Age'] >= 35 and df.loc[row, 'Age'] < 45:
            df.loc[row, 'Age Bracket'] = '35-45'
        elif df.loc[row, 'Age'] >= 45 and df.loc[row, 'Age'] < 55:
            df.loc[row, 'Age Bracket'] = '45-55'
        elif df.loc[row, 'Age'] >= 55 and df.loc[row, 'Age'] < 65:
            df.loc[row, 'Age Bracket'] = '55-65'
        elif df.loc[row, 'Age'] >= 65:
            df.loc[row, 'Age Bracket'] = '65+'
    # add Age Bracket to the question dictionary
    questions_df.loc[len(questions_df)] = ['Determined field broken into age brackets (0,25,35,45,55,65+)', 'Age Bracket']





###
### DATA VISUALIZATION ###
###



### Student Distributions for Characteristics and Numerical Survey Questions

    # Create distribution/bar charts to see variable distributions
    st.markdown('## Student Characteristic Distributions')

    # Format the visuals into columns
    cols = st.columns(2)
    
    with cols[0]:

        # Selectbox for student distribution plot
        # Create DataFrame for selected variables
        vars_df = df[['Gender','Education Level','Employment Status','Ethnicity/Race','Age Bracket']]
        # Streamlit dropdown to select the column to group by
        group_by_column = st.selectbox('Select a characteristic to see the student distribution: ', vars_df.columns, key="category_selectbox_1")

        # Group the data by the selected column and calculate the count (or any other aggregate)
        grouped_df = df.groupby(group_by_column, observed = False).size().reset_index(name='Count')

        # Sort the DataFrame by 'Count' (replace with the actual column name if it's different)
        grouped_df_count = grouped_df.sort_values(by='Count', ascending=False)

        # Create the first Plotly bar chart with a unique key
        fig1 = px.bar(grouped_df_count, x=group_by_column, y='Count')
        
        # Hide the xaxis lables
        fig1.update_xaxes(title_text=None)

        # Set a custom label for the y-axis
        fig1.update_layout(yaxis_title="Number of Students")

        # display the question from the survey associated with the figure
        question = questions_df[questions_df['New_Column_Title'] == group_by_column]['Original_Question'].values[0]
        st.write(f"**{question}**")
        
        # Show the first plot in Streamlit with a unique key
        st.plotly_chart(fig1, key='fig1_unique_key')

    with cols[1]:

        # Selectbox for variable count plots
        # Create DataFrame for selected variables
        vars_df_2 = df[['Attendance','Prior coding level','Instructor Satisfaction','Course Difficulty','Real World Preparation','Class Speed Rating','Motivation Level','Financial Stress Impact','Schedule Flexibility','Extra Practice Hours','Preferred Class Days']]
        # Streamlit dropdown to select the column to group by
        group_by_column_2 = st.selectbox('Select a measure to see the average by characteristic: ', vars_df_2.columns)

        # Group the DataFrame by the column and calculate the mean for each of the weeks
        grouped_df_avg = df.groupby(group_by_column, observed=False)[group_by_column_2].mean().reset_index()

        # Calculate the 25th and 75th percentiles for coloring
        lower_25th = grouped_df_avg[group_by_column_2].quantile(0.25)
        upper_75th = grouped_df_avg[group_by_column_2].quantile(0.75)

        # Assign colors based on the value (Green for top 25%, Red for bottom 25%, Gray for middle)
        grouped_df_avg['Color'] = grouped_df_avg[group_by_column_2].apply(
            lambda x: 'green' if x >= upper_75th else ('red' if x <= lower_25th else 'gray')
        )

        # Create a list of categories ordered by 'Count' in the first chart
        category_order = grouped_df_count[group_by_column].tolist()

        # Use a dictionary to map categories to their position in category_order
        category_position_map = {category: i for i, category in enumerate(category_order)}

        # Apply the position map to create an 'Order' column in grouped_df_avg
        grouped_df_avg['Order'] = grouped_df_avg[group_by_column].map(category_position_map)

        # Sort grouped_df_avg by 'Order' (preserving category labels)
        grouped_df_avg = grouped_df_avg.sort_values(by='Order')

        # Create the second Plotly bar chart with a unique key
        fig2 = px.bar(grouped_df_avg, x=group_by_column, y=group_by_column_2, color='Color', 
                    color_discrete_map={'green': 'green', 'red': 'red', 'gray': 'lightgray'})
        
        # Update the axes to show the category labels
        fig2.update_layout(xaxis_title=group_by_column,
                            yaxis_title=f'Average {group_by_column_2}',
                            xaxis_tickangle=45)
        
        # Hide the xaxis lables
        fig2.update_xaxes(title_text=None)

        # display the question from the survey associated with the figure
        question = questions_df[questions_df['New_Column_Title'] == group_by_column_2]['Original_Question'].values[0]
        st.write(f"**{question}**")

        # Show the second plot in Streamlit with a unique key
        st.plotly_chart(fig2, key='fig2_unique_key')
    



### Week to Week Attendance Visualization
    st.markdown('## Weekly Attendance Tracking') 

    # Create holder dataframe which gives numbers to the class attendance
    df_holder = df.copy()
    df_holder['WebDev_Fundamentals'] = df_holder['WebDev_Fundamentals'].replace({True: 1, False: 0})
    df_holder['Database_Fundamentals'] = df_holder['Database_Fundamentals'].replace({True: 1, False: 0})
    df_holder['Cloud_Services'] = df_holder['Cloud_Services'].replace({True: 1, False: 0})
    df_holder['Server_Side_Code'] = df_holder['Server_Side_Code'].replace({True: 1, False: 0})
    # Streamlit dropdown to select the column to group by
    group_by_column_2 = st.selectbox('Select a characteristic to see the student distribution: ', vars_df.columns, key="category_selectbox_2")
    # Group the DataFrame by the column and calculate the mean for each week
    grouped_df = df.groupby(group_by_column_2, observed=False)[['WebDev_Fundamentals', 'Database_Fundamentals', 'Cloud_Services', 'Server_Side_Code']].mean().reset_index()

    # Convert the grouped DataFrame to long format to use with Plotly
    long_df = grouped_df.melt(id_vars=[group_by_column_2], value_vars=['WebDev_Fundamentals', 'Database_Fundamentals', 'Cloud_Services', 'Server_Side_Code'],
                            var_name='Week', value_name='Average')

    # Define custom x-axis labels (replace 'Week 1', 'Week 2', etc. with your custom names)
    week_mapping = {
        'WebDev_Fundamentals': 'Week 1',
        'Database_Fundamentals': 'Week 2',
        'Cloud_Services': 'Week 3',
        'Server_Side_Code': 'Week 4',}

    # Apply the mapping to the 'Week' column in the dataframe
    long_df['Week'] = long_df['Week'].map(week_mapping)

    # Create a subplot for each category
    categories = grouped_df[group_by_column_2].unique()  # Get all unique categories
    num_categories = len(categories)

    # Determine rows and columns for subplots (e.g., if you have 4 categories, you want 2x2 subplots)
    cols = 2  # Number of columns in the subplot grid
    rows = (num_categories + cols - 1) // cols  # Calculate number of rows based on number of categories

    # Create subplots with a dynamic number of rows and columns
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f'Plot {i + 1}: {category}' for i, category in enumerate(categories)],
        vertical_spacing=0.1)

    # Add a bar trace for each category in a subplot
    for i, category in enumerate(categories):
        category_data = long_df[long_df[group_by_column_2] == category]
        
        row = (i // cols) + 1  # Determine the row position for the subplot
        col = (i % cols) + 1   # Determine the column position for the subplot

        fig.add_trace(go.Scatter(x=category_data['Week'], y=category_data['Average'], name='lines'),
                    row=row, col=col)
        
        # Hide x-axis labels for subplots above the last row (except for the bottom row)
        if row != rows:
            fig.update_xaxes(showticklabels=False, row=row, col=col)

    # Update layout for the final figure
    fig.update_layout(
        height=140 * rows,  # Adjust height based on the number of rows
        width=700,          # Adjust width
        title_text='',
        showlegend=False,   # Hide legend
        margin=dict(t=80, b=0, l=90, r=90))

    # Update the y-axis labels to show percentages with no decimal points
    fig.update_yaxes(tickformat=".0%")  # Format as percentage with no decimal places

    # Show the plot in Streamlit
    st.plotly_chart(fig)




### Multiple Choice question visualizations
    st.markdown('## Multiple Choice Survey Questions') 

    multi_options = {
        'Which programming languages were you already familiar with before this course?': ['Python', 'JavaScript', 'Java', 'C++', 'SQL', 'None', 'R'],
        'What was your main reason for enrolling in this course?': ['Personal Interest', 'Career Development', 'Skill Enhancement', 'Academic Requirement', 'Networking', 'Entrepreneurship'],
        'What factors influenced your decision to enroll in this course?': ['Affordable Cost', 'Interesting Topics', 'Job Support Opportunities', 'Hands On Practical Learning', 'Flexible Format', 'No Experience Required'],
        'What factors would encourage you to enroll in another course?': ['Lower Cost', 'Advanced Topics Skill Building', 'Stronger Job Support', 'Certification Provided', 'Flexible Scheduling Options', 'More Hands On Real World Experience']
        }
    
    # Streamlit dropdown to select the column to group by
    multi_selected_question = st.selectbox('Select a question to see the student distribution: ', multi_options.keys())

    # Get the relevant columns from the DataFrame
    selected_columns = multi_options[multi_selected_question]

    # Count True values in each selected column
    true_counts = df[selected_columns].sum().reset_index()
    true_counts.columns = ['Response', 'Count']

    # Sort the counts from highest to lowest
    true_counts = true_counts.sort_values(by='Count', ascending=False)

    # Create bar plot
    fig = px.bar(true_counts, x='Response', y='Count',
                labels={'Response': '', 'Count': 'Number of Responses'})

    # Rotate x-axis labels for readability
    fig.update_layout(xaxis_tickangle=45)

    # Display plot
    st.plotly_chart(fig)




### Visualize the TRUE/FALSE survey questions
    st.markdown('## TRUE/FALSE Survey Questions')  

    # Create a dictionary that maps full questions to column names
    question_map = {'Would you recommend this course to a friend?':'Recommend To Friend',
                    'Did you experience scheduling conflicts that made it difficult to attend class?':'Scheduling Conflicts',
                    'Did you face any technical barriers while taking this course?':'Technical Barriers',
                    'Would additional tutor or mentor support have been helpful for you?':'Tutor Support Need',
                    'Did you feel that you needed extra time to practice outside of class?':'Extra Practice Needed',
                    'Did you feel that a 90-minute class session allowed you to engage with the material at a comfortable pace and effectively absorb the content?':'Session Length Satisfaction'
    }

    # Create selectbox showing full questions, return the corresponding column name
    selected_question = st.selectbox('Select a TRUE/FALSE question to analyze: ', list(question_map.keys()))
    bool_column = question_map[selected_question]

    # Streamlit dropdown to select the column to group by (categorical)
    characteristic_df = df[['Gender','Education Level','Employment Status','Ethnicity/Race','Age Bracket']]
    group_by_column = st.selectbox('Select a characteristic to group by: ', characteristic_df.columns)

    # Group and count True/False values
    grouped_counts = df.groupby([group_by_column, bool_column], observed=False).size().reset_index(name='Count')

    # Rename columns for clarity in the plot
    grouped_counts[bool_column] = grouped_counts[bool_column].map({True: 'Yes', False: 'No'})

    # Wrap the title for better formatting
    #title_text = f'Distribution of True/False Responses for {bool_column} grouped by {group_by_column}'
    title_text = f'Distribution of True/False Responses for <b>{bool_column}</b> grouped by <b>{group_by_column}</b>'
    wrapped_title = "<br>".join(textwrap.wrap(title_text, width=90))

    # Create a stacked bar chart
    fig3 = px.bar(grouped_counts, x=group_by_column, y='Count', color=bool_column,
                title=wrapped_title, barmode='stack',
                color_discrete_map={'Yes': 'green', 'No': 'red'})

    # Update layout
    fig3.update_layout(xaxis_title='',
                    yaxis_title='Number of Students',
                    xaxis_tickangle=45)

    # Show the plot
    st.plotly_chart(fig3, key='fig3_unique_key')

    dataframe = df.copy()




### Where did you hear about SBCS?
    st.markdown('## Where did you hear about SBCS?')

    # List of columns to include for counting occurrences
    selected_columns = ['social_media', 'edu_org', 'community_org', 'network']

    # Combine the selected columns into a single series (ignoring other columns)
    combined_values = pd.concat([df[col] for col in selected_columns])

    # Count occurrences
    value_counts = combined_values.value_counts()

    # Plotting the bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    value_counts.plot(kind='bar', ax=ax, color='skyblue')

    # Set labels
    ax.set_ylabel('Number of Students')

    # Rotate x-axis labels to 45 degrees with correct alignment
    plt.xticks(rotation=45, ha='right', va='top')

    # Show the plot in Streamlit
    st.pyplot(fig)




### K Means Clustering 
    st.markdown('## Clustering Analysis')

    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer

    # Data cleaning
    drop_cols = ['Date of Birth', 'First_Name', 'Last_Name', 'Submit_Date', 'referral_source']
    df = df.drop(columns=drop_cols, errors='ignore')

    df = df.replace({'TRUE': True, 'FALSE': False})

    categorical_cols = ['Gender', 'Education Level', 'Employment Status', 'Ethnicity/Race']
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if 'Age' in df.columns:
        numerical_cols.append('Age')
    if 'Attendance' in df.columns:
        numerical_cols.append('Attendance')

    numerical_cols = list(set(numerical_cols))


    categorical_cols = [col for col in categorical_cols if col in df.columns]

    # Build preprocessing pipeline
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numerical_cols),
        
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first'))
        ]), categorical_cols)
    ])

    X_preprocessed = preprocessor.fit_transform(df)


    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans


    # Initialize session state for toggle if it doesn't exist yet
    if 'show_content' not in st.session_state:
        st.session_state.show_content = False

    # Button to toggle content visibility
    if st.button("Show/Hide Details on Choosing Number of Clusters"):
        st.session_state.show_content = not st.session_state.show_content

    # Conditional display
    if st.session_state.show_content:

        # Determine optimal K using Elbow Method and Silhouette Score
        wcss = []
        silhouette_scores = []
        K_range = range(2, 11)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X_preprocessed)
            wcss.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_preprocessed, labels))

        plt.figure(figsize=(12, 5))

        # Elbow plot
        plt.subplot(1, 2, 1)
        plt.plot(K_range, wcss, marker='o')
        plt.title('Elbow Method')
        plt.xlabel('K')
        plt.ylabel('WCSS')

        # Silhouette plot
        plt.subplot(1, 2, 2)
        plt.plot(K_range, silhouette_scores, marker='s', color='green')
        plt.title('Silhouette Score')
        plt.xlabel('K')
        plt.ylabel('Score')

        plt.tight_layout()
        plt.show()

        # Display the plots in Streamlit
        st.pyplot(plt)

        # Explain how to choose k_optimal
        st.markdown('To choose the optimal number of clusters for K-Means clustering, you can use the elbow method and the silhouette score. In the elbow graph, the number of clusters (k) is plotted against the within-cluster sum of squares (WCSS). The "elbow" point, where the rate of decrease sharply slows, suggests the optimal k—adding more clusters beyond this point yields diminishing returns. The silhouette score graph measures how well-separated the clusters are; the optimal k usually corresponds to the highest average silhouette score, indicating well-defined and distinct clusters. Together, these methods help balance cluster compactness and separation. Try to find a k value that is an elbow point and matches a peak in silhouette score.')

    # Ask the user to enter an integer value
    user_value = st.number_input("Enter the optimal number of clusers (integer value 2-10):", min_value=2, step=1, value=4, max_value=10)

    k_optimal = user_value
    kmeans = KMeans(n_clusters=k_optimal, random_state=42)
    labels = kmeans.fit_predict(X_preprocessed)

    df['cluster'] = labels

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_preprocessed)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50)

    plt.title('Learner Segmentation: Engagement vs Support Needs', fontsize=14)
    plt.xlabel('Tech Readiness & Learning Engagement')
    plt.ylabel('Support Needs & External Barriers')

    custom_labels = {}

    for cluster_id in range(k_optimal):
        # Get points for current cluster
        cluster_points = X_pca[labels == cluster_id]

        # Mean and std dev for ellipse size
        x_mean, y_mean = cluster_points[:, 0].mean(), cluster_points[:, 1].mean()
        x_std, y_std = cluster_points[:, 0].std(), cluster_points[:, 1].std()

        # Draw ellipse (1 std dev radius)
        ellipse = Ellipse(
            (x_mean, y_mean), width=2 * x_std, height=2 * y_std,
            edgecolor='black', linestyle='dotted', facecolor='none', linewidth=1.5
        )
        plt.gca().add_patch(ellipse)
        
        plt.text(
            x_mean, y_mean, custom_labels.get(cluster_id, f"Cluster {cluster_id}"),
            fontsize=10, weight='bold', ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )

    cbar = plt.colorbar(scatter)
    cbar.set_label('Cluster')

    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Display the plots in Streamlit
    st.pyplot(plt)  



### Clustering Analysis
    st.markdown("### Cluster Information")

# Numeric factor clustering visualizations
    # Define the top numerical features
    features = [
        'Age', 'Attendance', 'Python', 'SQL', 'Prior coding level',
        'Career Development', 'Personal Interest', 'Skill Enhancement',
        'Financial Stress Impact', 'Motivation Level'
    ]

    # Let the user select one numerical feature to visualize
    selected_numerical = st.selectbox("Select a numerical feature to view by cluster:", features)

    # Group by cluster and compute the mean of the selected feature
    numerical_means = df.groupby('cluster')[selected_numerical].mean().reset_index()

    # Calculate the overall average
    overall_average = df[selected_numerical].mean()

    # Create the bar chart with Plotly
    fig = px.bar(
        numerical_means,
        x='cluster',
        y=selected_numerical,
        title=f'Average {selected_numerical} by Cluster',
        labels={'cluster': 'Cluster', selected_numerical: f'Average {selected_numerical}'},
        text=selected_numerical
    )

    # Update bar style
    fig.update_traces(
        marker=dict(color='blue', opacity=0.3),
        textposition='outside',
        textfont=dict(color='black'),  # Keep labels solid and readable
        texttemplate='%{text:.2f}',
    )

    # Add horizontal line for overall average
    fig.add_hline(
        y=overall_average,
        line_dash='dash',
        line_color='black',
        opacity=0.5,
        annotation_text=f'Overall Avg: {overall_average:.2f}',
        annotation_position="top left",
        annotation_font_color="black"
    )

    # Update layout to avoid label cutoff
    fig.update_layout(
        xaxis_type='category',
        xaxis_tickangle=0,
        height=500,
        margin=dict(t=80, b=100),
    )

    # Ensure text labels are outside and readable
    fig.update_traces(
        texttemplate='%{text:.2f}',
        textposition='outside'
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig)


# Categorical factor clustering visualizations
    # Define categorical features
    categorical_features = ['Gender', 'Education Level', 'Employment Status', 'Ethnicity/Race']

    # Let the user choose a feature
    selected_feature = st.selectbox("Select a categorical feature to view by cluster:", categorical_features)

    # Only proceed if the selected feature exists in the DataFrame
    if selected_feature in df.columns:
        # Calculate the normalized value counts per cluster
        distribution = (
            df.groupby('cluster')[selected_feature]
            .value_counts(normalize=True)
            .rename('proportion')
            .reset_index()
        )

        # Create a stacked bar chart
        fig = px.bar(
            distribution,
            x='cluster',
            y='proportion',
            color=selected_feature,
            title=f'Distribution of {selected_feature} by Cluster',
            labels={'proportion': 'Proportion of Students', 'cluster': 'Cluster'},
            barmode='stack'
        )

        # Improve layout
        fig.update_layout(xaxis_type='category', xaxis_tickangle=0)
        st.plotly_chart(fig)