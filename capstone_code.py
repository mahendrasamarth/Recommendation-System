import streamlit as st
import pandas as pd
import numpy as np
import random
import pickle
from lightfm import LightFM
from lightfm.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

# Function to load data with caching to speed up app loading
@st.cache_data
def load_data():
    users = pd.read_csv('C:\\Users\\SAMARTH\\Desktop\\Capstone\\Capstone_Data\\users.csv')
    products = pd.read_csv('C:\\Users\\SAMARTH\\Desktop\\Capstone\\Capstone_Data\\products.csv')
    orders = pd.read_csv('C:\\Users\\SAMARTH\\Desktop\\Capstone\\Capstone_Data\\orders.csv')
    order_items = pd.read_csv('C:\\Users\\SAMARTH\\Desktop\\Capstone\\Capstone_Data\\order_items.csv')
    inventory_items = pd.read_csv('C:\\Users\\SAMARTH\Desktop\\Capstone\\Capstone_Data\\inventory_items.csv')
    final_df = pd.read_csv('C:\\Users\\SAMARTH\\Desktop\\Capstone\\final_df.csv')
    return users, products, orders, order_items, inventory_items, final_df

# Preparing the dataset
@st.cache_data
def prepare_data():
    data = pd.merge(users, orders, left_on='id', right_on='user_id',suffixes=('_users','_orders'))
    data = pd.merge(data, order_items, left_on='order_id', right_on='order_id',suffixes=('','_ord_items'))
    data = pd.merge(data, inventory_items, left_on='inventory_item_id', right_on='id',suffixes=('','_inv_items'))
    data = pd.merge(data, products, left_on='product_id', right_on='id',suffixes=('','_prod'))
    #Change: Filter out the data to remove cancelled orders from the recommendation
    #data = data[data['status'] != 'cancelled']
    # Dropping unnecessary columns
    data = data.drop(columns=['first_name', 'last_name', 'email', 'state', 'street_address', 'postal_code', 'city', 'country', 
                              'latitude', 'longitude', 'traffic_source', 'created_at_users', 'order_id', 'user_id', 'status', 
                              'gender_orders', 'created_at_orders', 'returned_at', 'shipped_at', 'delivered_at', 'num_of_item', 
                              'id_ord_items', 'user_id_ord_items','inventory_item_id', 'status_ord_items', 'created_at',
                               'shipped_at_ord_items', 'delivered_at_ord_items', 'returned_at_ord_items', 'id_inv_items', 
                               'product_id_inv_items', 'created_at_inv_items', 'sold_at', 'cost', 'product_category', 
                               'product_retail_price','product_sku', 'product_distribution_center_id', 'id_prod', 'cost_prod', 
                               'category', 'name', 'brand', 'retail_price', 'department', 'sku', 'distribution_center_id', 'cust_spend_segment'], errors='ignore')
    data.rename(columns={'id': 'user_id', 'gender_users': 'gender'}, inplace=True)
    return data

@st.cache_data
def load_model():
    with open('pre_train.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model
model = load_model()

# Your provided functions here
def get_preferred_brand(user_id, returned_products):
    purchased_products = order_items[order_items['user_id']==user_id]['product_id'].values
    purchased_brands = {}
    #print(returned_products, purchased_products)
    for i in purchased_products:
        #if i in returned_products:
        #    continue
        brand = products[products['id']==i]['brand'].values[0]
        if brand in purchased_brands:
            purchased_brands[brand] += 1
        else:
            purchased_brands[brand] = 1
    max_value = max(purchased_brands.values())
    max_keys = [key for key, value in purchased_brands.items() if value == max_value]
    return random.choice(max_keys)

def get_user_age_and_spending(user_id):
    personal_records = data[data['user_id']==user_id]
    age = personal_records['age'].values[0]
    gender = personal_records['gender'].values[0]
    avg_spending = personal_records['sale_price'].sum()/personal_records['sale_price'].count()
    max_spent = personal_records['sale_price'].max()
    min_spent = personal_records['sale_price'].min()
    return age, gender, avg_spending, max_spent, min_spent

def get_returned_products(user_id):
    return order_items[(order_items['user_id']==user_id) & (order_items['status'].isin(['Cancelled', 'Returned']))]['product_id'].values

def get_best_sellers(num_recommendations=5):
    total_sales_per_product = order_items.groupby('product_id').size()
    top_selling_products = total_sales_per_product.sort_values(ascending=False).head(num_recommendations)
    return top_selling_products.index.tolist()

def get_new_arrivals(num_recommendations=5):
    inventory_data_sorted = inventory_items.sort_values(by='created_at', ascending=False)
    return inventory_data_sorted.head(num_recommendations)['product_id'].tolist()

def get_product_gender_preference(product_id):
    gender = inventory_items[inventory_items['product_id']==product_id]['product_department'].values[0]
    return 'M' if gender=='Men' else 'F'

def get_product_brand(product_id):
    return inventory_items[inventory_items['product_id']==product_id]['product_brand'].values[0]

def get_purchased_products(user_id):
    pur_prod = data[data.user_id == user_id]
    return pur_prod[['product_name', 'product_id', 'sale_price', 'product_brand']], pur_prod['product_id'].values  # Ensure this is a DataFrame

def get_product_price(product_id):
    return data[data['product_id']==product_id]['sale_price'].values[0]

def get_product_details_to_display(ids):
    data_list = []
    for id in ids:
        prod = products[products['id']==id]
        #status = final_df[final_df['product_id']==id]['status_item'].values[0]
        status = order_items[order_items['product_id']==id]['status'].values[0]
        data_f = {
            #'Product ID': id,
            'Name': prod['name'].values[0],
            'Retail Price': prod['retail_price'].values[0].round(2),
            'Brand': get_product_brand(id).title(),
            'Gender Preference': get_product_gender_preference(id),
            'Status': status,
            'Category': prod['category'].values[0]
        }
        # Append the values to the list
        data_list.append(data_f)
        # Convert the list of dictionaries to a DataFrame
    result_df = pd.DataFrame(data_list)
    # Adjust index to start from 1
    result_df.index = result_df.index + 1
    return result_df

def get_recommendations(model, dataset, user_ids, num_recommendations=5, max_preferred_brand=3, spending_range=(0.8, 1.2)):
    n_users, n_items = dataset.interactions_shape()
    recommend, output = [], []
    product_id_mapping = list(dataset.mapping()[2].keys())  # Prepare product ID mapping once

    for user_id in user_ids:
        if user_id in users['id'].values.flatten():
            user_index = dataset.mapping()[0][user_id]
            scores = model.predict(user_index, np.arange(n_items))
            _, gender, avg_spending, max_spent, min_spent = get_user_age_and_spending(user_id)
            returned_products = set(get_returned_products(user_id))  # Ensuring this is a set
            preferred_brand = get_preferred_brand(user_id, returned_products)
            purchased_products = set(get_purchased_products(user_id)[0]['product_id']) # Convert to set for product IDs

            # Map scores to product IDs, excluding returned, purchased, and previously interacted products
            product_scores = {
                product_id_mapping[i]: scores[i] for i in range(n_items) 
                if product_id_mapping[i] not in returned_products 
                and product_id_mapping[i] not in purchased_products
            }
            output.append(product_scores)  # Store product scores for output

            # Create a sorted list of products based on scores, filtering by gender and excluding undesired products
            valid_products = [
                (prod, score) for prod, score in product_scores.items()
                if get_product_gender_preference(prod) == gender and prod not in returned_products
            ]
            valid_products.sort(key=lambda x: x[1], reverse=True)

            brand_recommendations = []
            other_recommendations = []

            for prod, score in valid_products:
                if len(brand_recommendations) + len(other_recommendations) == num_recommendations:
                    break
                product_price = get_product_price(prod)

                if (min_spent * spending_range[0] <= product_price <= max_spent * spending_range[1]):
                    if get_product_brand(prod) == preferred_brand and len(brand_recommendations) < max_preferred_brand:
                        brand_recommendations.append(prod)
                    elif len(other_recommendations) < (num_recommendations - max_preferred_brand):
                        other_recommendations.append(prod)
            
            # Fill in additional products if there are slots left
            if len(brand_recommendations) + len(other_recommendations) < num_recommendations:
                fill_count = num_recommendations - (len(brand_recommendations) + len(other_recommendations))
                additional_products = [prod for prod, _ in valid_products if prod not in brand_recommendations and prod not in other_recommendations][:fill_count]
                other_recommendations.extend(additional_products)

            final_recommendations = brand_recommendations + other_recommendations
            recommend.append(final_recommendations)
            #print(f'Recommended products for {user_id}:', [brand_recommendations + other_recommendations])
        else:
            new_arrived = get_new_arrivals(num_recommendations//2+1)
            best_seller = get_best_sellers(num_recommendations//2+1)
            final_recommendations = new_arrived + best_seller
            new_user_recommendations = np.random.choice(final_recommendations, size=num_recommendations, replace=False).tolist()
            recommend.append(new_user_recommendations)
            #print('New User Recommendations', new_user_recommendations)
    return recommend, output

# Function to vectorize product descriptions and compute similarity
@st.cache_resource
def get_vectorizer_and_matrix(descriptions):
    descriptions = descriptions.fillna("missing").astype(str)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    return vectorizer, tfidf_matrix

# def search_products(query, vectorizer, tfidf_matrix):
#     query = str(query)
#     query_vector = vectorizer.transform([query])
#     cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
#     similar_indices = cosine_similarities.argsort()[-5:][::-1]  # Top 5 results
#     return products.iloc[similar_indices]

def search_products(query, vectorizer, tfidf_matrix):
    query = str(query)
    query_vector = vectorizer.transform([query])
    cosine_similarities = linear_kernel(query_vector, tfidf_matrix).flatten()
    
    # Get top_n indices of most similar products
    similar_indices = cosine_similarities.argsort()[-5 * 2:][::-1]  # Get more results for re-ranking
    
    # Re-rank based on the number of query terms present
    re_ranked_indices = []
    query_terms = set(query.split())
    for index in similar_indices:
        product_terms = set(products.iloc[index]['name'].split())
        common_terms_count = len(query_terms.intersection(product_terms))
        re_ranked_indices.append((index, common_terms_count))
    
    # Sort based on common term count and take the top_n
    re_ranked_indices.sort(key=lambda x: x[1], reverse=True)
    top_indices = [index for index, count in re_ranked_indices[:5]]
    
    return products.iloc[top_indices]


@st.cache_data
def get_top_and_bottom_brands_by_segment(data):
    # Group by 'Quartile' and 'brand', count occurrences, and reset index
    brand_counts = data.groupby(['Quartile', 'brand'], observed=True).size().reset_index(name='counts')
    
    # Sort the data within each quartile by counts in descending order
    brand_counts_sorted = brand_counts.sort_values(['Quartile', 'counts'], ascending=[True, False])
    
    # Function to get top 5 and bottom 2 entries per group
    def get_top_bottom_brands(group):
        return pd.concat([
            group.head(5),  # Top 5 entries
            group.tail(2)   # Bottom 2 entries
        ])

    # Apply function to each quartile group
    top_bottom_brands = brand_counts_sorted.groupby('Quartile', observed=False).apply(get_top_bottom_brands).reset_index(drop=True)
    top_bottom_brands['brand'] = top_bottom_brands['brand']
    # Resetting the index to start from 1
    top_bottom_brands.reset_index(drop=True, inplace=True)
    top_bottom_brands.index += 1
    
    # Renaming columns for clarity
    top_bottom_brands.rename(columns={
        'Quartile': 'Customer Spend Segment',
        'brand': 'Brand',
        'counts': 'Total Purchases'
    }, inplace=True)
    
    return top_bottom_brands

@st.cache_data
def get_top_category_by_age_group(data):
    # Group by 'Age Group' and 'category', and count occurrences
    category_counts = data.groupby(['age_group', 'category']).size().reset_index(name='counts')
    
    # Find the category with the maximum count in each age group
    top_categories = category_counts.loc[category_counts.groupby('age_group', observed=False)['counts'].idxmax()]
    
    # Resetting the index to start from 1
    top_categories.reset_index(drop=True, inplace=True)
    top_categories.index += 1

    # Renaming columns for better readability
    top_categories.rename(columns={'age_group': 'Age Group','category': 'Top Category', 'counts': 'Total Counts'}, inplace=True)
    return top_categories

users, products, orders, order_items, inventory_items, final_df = load_data()
age_group_order = ['Young', 'Middle-aged', 'Elderly', 'Senior']
final_df['age_group'] = pd.Categorical(final_df['age_group'], categories=age_group_order, ordered=True)
final_df['Max_Value'] = final_df.groupby('user_id')['sale_price'].transform('max')
# Categorizing into quartiles
final_df['Quartile'] = pd.qcut(final_df['Max_Value'], 4, labels=['Low Spender', 'Below Average Spender', 'Above Average Spender', 'Premium Spender'])

data = prepare_data()
# Prepare the interaction data
interaction_data = data.groupby(['user_id', 'product_id']).size().reset_index(name='interaction_count')

dataset = Dataset()
dataset.fit((row['user_id'] for index, row in interaction_data.iterrows()),
            (row['product_id'] for index, row in interaction_data.iterrows()))

(interactions, weights) = dataset.build_interactions(((row['user_id'], row['product_id']) for index, row in interaction_data.iterrows()))

# Prepare vectorizer and TF-IDF matrix
vectorizer, tfidf_matrix = get_vectorizer_and_matrix(products['name'])

# Streamlit UI starts here
st.title('Product Recommendation System')

# Streamlit UI code for product search
st.sidebar.header("Product Search")
search_query = st.sidebar.text_input("Enter keywords to search for products:", "")

if search_query:
    with st.spinner("Searching for products..."):
        result = search_products(search_query, vectorizer, tfidf_matrix)
        if not result.empty:
            st.write("### Search Results:")
            for _, row in result.iterrows():
                st.write(f"**{row['name']}**")
        else:
            st.write("No products found matching your search.")

# Sidebar for user input
with st.sidebar:
    st.header("User Input")
    # Add a radio button to select user type
    user_type = st.radio("Select User Type:", ('Existing User', 'New User'))
    
    # Conditionally display the User ID input
    if user_type == 'Existing User':
        user_id_input = st.text_input("Enter User ID:")
    else:
        user_id_input = None  # Set to None if New User is selected
    
    st.header("Actions")

    # Button to refresh data
    # if st.button("Refresh Data"):
    #     #st.clear_cache()
    #     st.rerun()

    if st.button("Show User Statistics"):
        user_count = users['id'].nunique()
        average_orders = orders.groupby('user_id').size().mean()
        st.session_state['Total Users'] = user_count
        st.session_state['Average Order'] = average_orders

    if st.button("List Fav Brands"):
        favorite_brands_by_segment = get_top_and_bottom_brands_by_segment(final_df)
        # This will store the DataFrame in session state to display in main window
        st.session_state['favorite_brands_by_segment'] = favorite_brands_by_segment
    
    if st.button("Top Category by each Age Group"):
        st.session_state['top_category'] = 0

# Main window display logic
if 'favorite_brands_by_segment' in st.session_state:
    st.write("### Favorite Brands by Customer Spend Segment:")
    st.dataframe(st.session_state['favorite_brands_by_segment'])

if 'Total Users' in st.session_state and 'Average Order' in st.session_state:
    st.write("### User Statistics")
    st.write(f"Total Users: {st.session_state['Total Users']}")
    st.write(f"Average Orders per User: {st.session_state['Average Order']:.2f}")

if 'top_category' in st.session_state:
    top_categories_by_age = get_top_category_by_age_group(final_df)
    st.write("### Top Categories by Age Group:")
    st.dataframe(top_categories_by_age)

# Logic to process inputs based on user type
if user_type == 'Existing User' and user_id_input:
    if user_id_input.isdigit():
        user_id = int(user_id_input)
        user_ids = [user_id]

        with st.spinner('Calculating Recommendations...'):
            recommendations, _ = get_recommendations(model, dataset, user_ids)

        st.subheader("Customer Details")
        if user_id in users['id'].values:
            user_details = users[users['id'] == user_id]
            return_products = get_returned_products(user_id)
            customer_name = f"{user_details['first_name'].values[0]} {user_details['last_name'].values[0]}"
            st.markdown(f"##### Name: {customer_name}")
            age, gender, avg_spending, max_spent, min_spent = get_user_age_and_spending(user_id)
            #st.write(f"Age: {age} Gender: {gender} Average Spending: ${avg_spending:.2f} Maximum Spent: ${max_spent:.2f} Minimum Spent: ${min_spent:.2f}")
            segment = final_df[final_df['user_id']==user_id]['Quartile'].max()
            st.markdown(f"""
                    - **Age:** {age}
                    - **Gender:** {gender}
                    - **Average Spending:** ${avg_spending:.2f}
                    - **Maximum Spent:** ${max_spent:.2f}
                    - **Minimum Spent:** ${min_spent:.2f}
                    - **Customer Spend Segment:** {segment}
                    """)
            st.write("Preferred Brand:", get_preferred_brand(user_id, return_products).title())
            st.write("---")

            purchased_products_df, product_ids = get_purchased_products(user_id)
            if not purchased_products_df.empty:
                st.subheader("Bought Products:")
                st.dataframe(get_product_details_to_display(product_ids))
            else:
                st.write("No products bought yet.")
        else:
            st.error("Invalid User ID. Please enter a valid User ID.")

        st.subheader("Recommendations")
        for user_id, user_recommendations in zip(user_ids, recommendations):
            st.write(f'Recommended Products for User ID: {user_id}')
            product_details_list = []

            for product_id in user_recommendations:
                product_info = products[products['id']==product_id]
                product_data = {
                    'Name': product_info['name'].values[0],
                    'Retail Price': product_info['retail_price'].values[0].round(2),
                    'Brand': get_product_brand(product_id),
                    'Gender Preference': get_product_gender_preference(product_id),
                    'Category': product_info['category'].values[0]
                }
                product_details_list.append(product_data)

            recommended_products_df = pd.DataFrame(product_details_list)
            # Adjust index to start from 1
            recommended_products_df.index = recommended_products_df.index + 1
            st.write(recommended_products_df)
    else:
        st.error("User ID should contain digits only. Please enter a valid User ID.")
elif user_type == 'New User':
    with st.spinner('Generating Recommendations for New Users...'):
        user_ids = [34985734]
        new_user_recommendations, _ = get_recommendations(model, dataset, user_ids, num_recommendations=5)
        st.subheader("Recommendations for New User")
        if new_user_recommendations:
            for user_id, user_recommendations in zip(user_ids, new_user_recommendations):
                product_details_list = []

                for product_id in user_recommendations:
                    product_info = products[products['id']==product_id]
                    product_data = {
                        'Name': product_info['name'].values[0],
                        'Retail Price': product_info['retail_price'].values[0].round(2),
                        'Brand': get_product_brand(product_id).title(),
                        'Gender Preference': get_product_gender_preference(product_id),
                        'Category': product_info['category'].values[0]
                    }
                    product_details_list.append(product_data)

            recommended_products_df = pd.DataFrame(product_details_list)
            st.write(recommended_products_df)
        else:
            st.write("No recommendations available at the moment.")
        
else:
    st.info("Please select a user type to proceed with recommendations.")

