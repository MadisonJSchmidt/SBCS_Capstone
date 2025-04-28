import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


st.title('South Bend Code School')
st.text('Coding Courses for Adult Learners')

st.markdown('## Student Body Demographics')

uploaded_file = st.file_uploader('Upload your file here.')

if uploaded_file:
    df = pd.read_excel(uploaded_file)



# Data Cleaning

    # save as strings
    df['Gender'] = df['Gender'].astype(str)
    df['Veteran Status'] = df['Veteran (Y/N)'].astype(str)
    df['Employment Status'] = df['Employment Status'].astype(str)
    df['Race/Ethnicity'] = df['Ethnicity'].astype(str)

    # fix the absence columns
    for row in range(0, len(df)):
        # change written values to indicate absence (0)
        if df.loc[row, 'Week 1'] != 0 | 1:
            df.loc[row, 'Week 1'] = 0
        if df.loc[row, 'Week 2'] != 0 | 1:
            df.loc[row, 'Week 2'] = 0
        if df.loc[row, 'Week 3'] != 0 | 1:
            df.loc[row, 'Week 3'] = 0
        if df.loc[row, 'Week 4'] != 0 | 1:   
            df.loc[row, 'Week 4'] = 0        
        # remove ages that are wrong
        if df.loc[row, 'Age'] < 16:
            df.loc[row, 'Age'] = None
        # Fix the columns to correct for capitalization
        df.loc[row, 'Gender'] = df.loc[row, 'Gender'].lower()
        df.loc[row, 'Veteran Status'] = df.loc[row, 'Veteran Status'].lower()
        df.loc[row, 'Employment Status'] = df.loc[row, 'Employment Status'].lower()
        df.loc[row, 'Race/Ethnicity'] = df.loc[row, 'Race/Ethnicity'].lower()
        df.loc[row, 'Race/Ethnicity'] = df.loc[row, 'Race/Ethnicity'].title()
        # Fix potential spelling errors in Employment Status
        if df.loc[row, 'Employment Status'] == 'studnet':
            df.loc[row, 'Employment Status'] = 'student'
        elif df.loc[row, 'Employment Status'] == 'full':
            df.loc[row, 'Employment Status'] = 'full-time'
        elif df.loc[row, 'Employment Status'] == 'retrired':
            df.loc[row, 'Employment Status'] = 'retired'
        elif df.loc[row, 'Employment Status'] == 'not':
            df.loc[row, 'Employment Status'] = 'unemployed'
        elif df.loc[row, 'Employment Status'] == 'part time':
            df.loc[row, 'Employment Status'] = 'part-time'
        elif df.loc[row, 'Employment Status'] == 'full time':
            df.loc[row, 'Employment Status'] = 'full-time'
        elif df.loc[row, 'Employment Status'] == 'nan':
            df.loc[row, 'Employment Status'] = 'other'
        # Fix potential spelling errors in Gender
        if df.loc[row, 'Gender'] == 'males':
            df.loc[row, 'Gender'] = 'male'
        elif df.loc[row, 'Gender'] == 'females':
            df.loc[row, 'Gender'] = 'female'
        elif df.loc[row, 'Gender'] == 'nan':
            df.loc[row, 'Gender'] = 'other'
        elif df.loc[row, 'Gender'] == 'prefer to self-describe':
            df.loc[row, 'Gender'] = 'other'
        # Combine ethnicities
        if df.loc[row, 'Race/Ethnicity'] == 'Hispanic':
            df.loc[row, 'Race/Ethnicity'] = 'Hispanic/Latino'
        elif df.loc[row, 'Race/Ethnicity'] == 'American Indian':
            df.loc[row, 'Race/Ethnicity'] = 'American Indian/Alaska Native'
        elif df.loc[row, 'Race/Ethnicity'] == 'Black':
            df.loc[row, 'Race/Ethnicity'] = 'Black/African American'
        elif df.loc[row, 'Race/Ethnicity'] == 'Nan':
            df.loc[row, 'Race/Ethnicity'] = 'Other'
        # change Veteran status nan to 'unanswered'
        if df.loc[row, 'Veteran Status'] == 'nan':
            df.loc[row, 'Veteran Status'] = 'not answered'

    # Change all attendance values to integers
    df['Week 1'] = df['Week 1'].astype(int)
    df['Week 2'] = df['Week 2'].astype(int)
    df['Week 3'] = df['Week 3'].astype(int)
    df['Week 4'] = df['Week 4'].astype(int)      
    
    # save cateogorical variables
    df['Gender'] = df['Gender'].astype('category')
    df['Ethnicity'] = df['Race/Ethnicity'].astype('category')
    df['Veteran Status'] = df['Veteran Status'].astype('category')
    df['Employment Status'] = df['Employment Status'].astype('category')

    # create a month-run column
    df['run_month'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Temp_Month'] + '-01')


# Visulaizations



    # Distribution Visualizations

    # Create DataFrame
    vars_df = df[['Race/Ethnicity','Gender','Employment Status','Veteran Status']]
    # Streamlit dropdown to select the column to group by
    group_by_column = st.selectbox('Select a trait to see the student distribution: ', vars_df.columns)
    # Group the data by the selected column and calculate the count (or any other aggregate)
    grouped_df = df.groupby(group_by_column, observed = False).size().reset_index(name='Count')

    # Format the visuals into columns
    cols = st.columns(2)
    
    with cols[0]:

        # Sort the DataFrame by 'Count' (replace with the actual column name if it's different)
        grouped_df_count = grouped_df.sort_values(by='Count', ascending=False)

        # Create the first Plotly bar chart with a unique key
        fig1 = px.bar(grouped_df_count, x=group_by_column, y='Count', title=f'Distribution by {group_by_column}')
        # Show the first plot in Streamlit with a unique key
        st.plotly_chart(fig1, key='fig1_unique_key')

    with cols[1]:

        # Group the DataFrame by the column and calculate the mean for each of the weeks
        grouped_df_avg = df.groupby(group_by_column, observed=False)[['Week 1', 'Week 2', 'Week 3', 'Week 4']].mean().reset_index()

        # Calculate the combined average of all four weeks (mean across all weeks for each group)
        grouped_df_avg['Combined Average'] = grouped_df_avg[['Week 1', 'Week 2', 'Week 3', 'Week 4']].mean(axis=1)

        # Calculate the 25th and 75th percentiles for coloring
        lower_25th = grouped_df_avg['Combined Average'].quantile(0.25)
        upper_75th = grouped_df_avg['Combined Average'].quantile(0.75)

        # Assign colors based on the value (Green for top 25%, Red for bottom 25%, Gray for middle)
        grouped_df_avg['Color'] = grouped_df_avg['Combined Average'].apply(
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
        fig2 = px.bar(grouped_df_avg, x=group_by_column, y='Combined Average', color='Color', 
                    color_discrete_map={'green': 'green', 'red': 'red', 'gray': 'lightgray'},
                    title=f'Average Attendance % by {group_by_column}')
        
        # Update the y-axis labels to show percentages with no decimal points
        fig2.update_yaxes(tickformat=".0%", title_text="Attendance Percentage")  # Format as percentage with no decimal places

        # Show the second plot in Streamlit with a unique key
        st.plotly_chart(fig2, key='fig2_unique_key')


    
    # Attendance Tracking for Different Categories

    # Group the DataFrame by the column and calculate the mean for each week
    grouped_df = df.groupby(group_by_column, observed=False)[['Week 1', 'Week 2', 'Week 3', 'Week 4']].mean().reset_index()

    # Convert the grouped DataFrame to long format to use with Plotly
    long_df = grouped_df.melt(id_vars=[group_by_column], value_vars=['Week 1', 'Week 2', 'Week 3', 'Week 4'],
                            var_name='Week', value_name='Average')

    # Create a subplot for each category
    categories = grouped_df[group_by_column].unique()  # Get all unique categories
    num_categories = len(categories)

    # Determine rows and columns for subplots (e.g., if you have 4 categories, you want 2x2 subplots)
    cols = 2  # Number of columns in the subplot grid
    rows = (num_categories + cols - 1) // cols  # Calculate number of rows based on number of categories

    # Create subplots with a dynamic number of rows and columns
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f'Plot {i + 1}: {category}' for i, category in enumerate(categories)],
        vertical_spacing=0.15  # Adjust spacing between rows
    )

    # Add a bar trace for each category in a subplot
    for i, category in enumerate(categories):
        category_data = long_df[long_df[group_by_column] == category]
        
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
        title_text=f'Attendance % by {group_by_column} Each Week',
        showlegend=False,   # Hide legend
        margin=dict(t=80, b=0, l=90, r=90))

    # Update the y-axis labels to show percentages with no decimal points
    fig.update_yaxes(tickformat=".0%")  # Format as percentage with no decimal places

    # Show the plot in Streamlit
    st.plotly_chart(fig)



    # Student Age Histogram

    # Create the Plotly histogram
    fig = px.histogram(df, x='Age', nbins=7, title='Student Age Distribution')
    # Update axes labels
    fig.update_layout(
        xaxis_title='Age',
        yaxis_title='Number of Students'
    )
    # Display the histogram in Streamlit
    st.plotly_chart(fig, use_container_width=True)



    # Attendance Percentage by Month

    # Group the DataFrame by 'run_month' and calculate the mean for each week
    grouped_df = df.groupby('run_month')[['Week 1', 'Week 2', 'Week 3', 'Week 4']].mean().reset_index()
    # Convert the grouped DataFrame to a long format (for use with Plotly)
    long_df = grouped_df.melt(id_vars=['run_month'], value_vars=['Week 1', 'Week 2', 'Week 3', 'Week 4'],
                          var_name='Week', value_name='Average')
    # Create the Plotly line chart
    fig = px.line(long_df, x='run_month', y='Average', color='Week', 
              title='Average Attendance Percentage by Run Month for Each Session')
    # Change the x-axis label
    fig.update_xaxes(title_text='Run Month')  # Change x-axis label to 'Run Month'
    # Update the y-axis labels to show percentages
    fig.update_yaxes(tickformat=".0%", title_text="Attendance Percentage")  # Format as percentage with no decimal places
    # Show the plot in Streamlit
    st.plotly_chart(fig)