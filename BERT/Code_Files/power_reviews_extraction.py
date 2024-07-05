import re
import os
import sys
import csv
import json
import time
import requests
import datetime


class PowerReviews(object):
    def __init__(self):
        self.merchant_id = 677699
        #api_key = "7601cd87-cad0-4e93-b529-8d44a4bae1af"
        self.api_key = "480b2c96-26e8-4675-abe4-603abd90edd1"
        # Date range can be 12 months, 6 months, 3 months, 30 days, 7 days, 2 days, last day, yesterday, today, this month
        #date_range = "12 months"
        
        if sys.platform.startswith("win"):
            self.path = os.getcwd() + "/../Data_Files/"
            self.time_file = self.path + 'input_time.csv'
        elif sys.platform.startswith("darwin"):
            self.path = os.getcwd() + "//Data_Files/"
            self.time_file = self.path + 'input_time.csv'
        elif sys.platform.startswith("linux"):
            self.path = os.getcwd() + "/../Data_Files/"
            self.time_file = self.path + 'input_time.csv'
    
    def read_time(self):
        fo = open(self.time_file, "r")
        data = fo.readlines()
        for line in data:
            input_time = line.strip()
        fo.close()
        return input_time
    
    def get_time_range(self):
        date_range = self.read_time()

        if re.search(r".*[a-zA-Z].*", date_range):
            get_date_range = date_range.upper()
            if get_date_range in ["12 MONTHS", "LAST 12 MONTHS"]:
                calculate_date = datetime.datetime.today() - datetime.timedelta(days=365)
            elif get_date_range in ["6 MONTHS", "LAST 6 MONTHS"]:
                calculate_date = datetime.datetime.today() - datetime.timedelta(days=180)
            elif get_date_range in ["3 MONTHS", "LAST 3 MONTHS"]:
                calculate_date = datetime.datetime.today() - datetime.timedelta(days=90)
            elif get_date_range in ["30 DAYS", "LAST 30 DAYS"]:
                calculate_date = datetime.datetime.today() - datetime.timedelta(days=30)
            elif get_date_range in ["7 DAYS", "LAST 7 DAYS"]:
                calculate_date = datetime.datetime.today() - datetime.timedelta(days=7)
            elif get_date_range in ["2 DAYS", "LAST 2 DAYS"]:
                calculate_date = datetime.datetime.today() - datetime.timedelta(days=2)
            elif get_date_range in ["1 DAY", "1 DAYS", "ONE DAY", "ONE DAYS", "LAST DAY", "YESTERDAY"]:
                calculate_date = datetime.datetime.today() - datetime.timedelta(days=1)
            elif get_date_range == "TODAY":
                calculate_date = datetime.datetime.today()
            elif get_date_range == "THIS MONTH":
                input_dt = datetime.datetime.today()
                first_day_of_month = input_dt.replace(day=1)
                calculate_date = first_day_of_month
            else:
                print("Invalid date. Please choose from the date range selection only")
                return False
            
            unix_time = int(calculate_date.timestamp())
            return str(unix_time)
        else:
            unix_time = int(time.mktime(datetime.datetime.strptime(date_range, "%d/%m/%Y").timetuple()))
            return str(unix_time)

    def url_formation(self):
        get_time = self.get_time_range()
        # Fetch the first page initially
        #review_data_url = f"https://readservices-b2c.powerreviews.com/m/{merchant_id}/reviews?apikey={api_key}&paging.from=1&paging.size=25&date={get_time}&sort=Newest"
        review_data_url = f"https://readservices-b2c.powerreviews.com/m/{self.merchant_id}/reviews?apikey={self.api_key}&paging.from=1&paging.size=25&date={get_time}&filters=&search=&sort=Newest&image_only=false&page_locale=en_GB"
        review_data = requests.get(review_data_url)
        data_bytes = review_data.content
        my_json_dict = json.loads(data_bytes)
        # Fetch total no. of pages
        # Disabling it for now since API doesn't work as expected and keeping value static to 25 below (approx 2 pages)
        fetch_total_pages = 25 
        # fetch_total_pages = int(my_json_dict['paging']['pages_total'])
        # print(f"total pages: {fetch_total_pages}")
        total_urls = []
        for pages in range(1, fetch_total_pages, 25):
            if pages == 1:
                #review_data_url = f"https://readservices-b2c.powerreviews.com/m/{merchant_id}/reviews?apikey={api_key}&paging.from={pages}&paging.size=25&date={get_time}&sort=Newest"
                review_data_url = f"https://readservices-b2c.powerreviews.com/m/{self.merchant_id}/reviews?apikey={self.api_key}&paging.from={pages}&paging.size=25&date={get_time}&filters=&search=&sort=Newest&image_only=false&page_locale=en_GB"
                total_urls.append(review_data_url)
            else:
                #review_data_url = f"https://readservices-b2c.powerreviews.com/m/{merchant_id}/reviews?apikey={api_key}&paging.from={pages - 1}&paging.size=25&date={get_time}&sort=Newest"
                review_data_url = f"https://readservices-b2c.powerreviews.com/m/{self.merchant_id}/reviews?apikey={self.api_key}&paging.from={pages - 1}&paging.size=25&date={get_time}&filters=&search=&sort=Newest&image_only=false&page_locale=en_GB"
                total_urls.append(review_data_url)
        return total_urls
    
    def clear_csv(self):
        if sys.platform.startswith("win"):
            power_reviews_data_path = os.getcwd() + "/../Data_Files/power_reviews_all_api.csv"
        elif sys.platform.startswith("darwin"):
            power_reviews_data_path = os.getcwd() + "/Data_Files/power_reviews_all_api.csv"
        elif sys.platform.startswith("linux"):
            power_reviews_data_path = os.getcwd() + "/../Data_Files/power_reviews_all_api.csv"

        fo = open(power_reviews_data_path, "w")
        fo.writelines("")
        fo.close()

    def write_csv_data(self, file_name, review_created_date, page_id, review_rating, review_headline, comment, review_location, reviewer_nickname):
        if sys.platform.startswith("win"):
            power_reviews_data_path = os.getcwd() + "/../Data_Files/power_reviews_all_api.csv"
        elif sys.platform.startswith("darwin"):
            power_reviews_data_path = os.getcwd() + "/Data_Files/power_reviews_all_api.csv"
        elif sys.platform.startswith("linux"):
            power_reviews_data_path = os.getcwd() + "/../Data_Files/power_reviews_all_api.csv"
        
        file_name = power_reviews_data_path
        try:
            with open(file_name, 'a', encoding = 'utf8', newline = '') as csvfile:
                fieldnames = ['Created Date', 'Page ID', 'Review Rating',
                                'Review Headline', 'Comment', 'Review Location',
                                'Reviewer Nickname']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                is_file_empty = os.stat(file_name).st_size
                if is_file_empty == 0:
                    writer.writeheader()
                writer.writerow({'Created Date': review_created_date, 'Page ID': page_id,
                                'Review Rating': review_rating,
                                'Review Headline': review_headline,'Comment': comment,
                                'Review Location': review_location,
                                'Reviewer Nickname': reviewer_nickname})
                csvfile.close()
        except:
            print(f"Something unexpected happened. Please check the output.")
    
    def get_info(self):
        if sys.platform.startswith("win"):
            power_reviews_data_path = os.getcwd() + "/../Data_Files/power_reviews_all_api.csv"
        elif sys.platform.startswith("darwin"):
            power_reviews_data_path = os.getcwd() + "/Data_Files/power_reviews_all_api.csv"
        elif sys.platform.startswith("linux"):
            power_reviews_data_path = os.getcwd() + "/../Data_Files/power_reviews_all_api.csv"

        file_name = power_reviews_data_path
        # Clearing existing CSV everytime before running script
        self.clear_csv()
        get_urls = self.url_formation()
        for url_num in range(len(get_urls)):
            get_json = requests.get(get_urls[url_num])
            get_dict = json.loads(get_json.content)
            reviews_dict = get_dict['results'][0]
            find_reviews_list = reviews_dict['reviews']
            for reviews_dict in find_reviews_list:
                get_unix_time = int(reviews_dict['details']['created_date']) / 1000
                get_date = datetime.datetime.fromtimestamp(get_unix_time).strftime('%c')
                self.write_csv_data(file_name, get_date, reviews_dict['details']['product_page_id'],
                                reviews_dict['metrics']['rating'], reviews_dict['details']['headline'],
                                reviews_dict['details']['comments'], reviews_dict['details']['location'],
                                reviews_dict['details']['nickname'])
        print("Power Reviews fetched successfully. Please check power_reviews_all_api.csv")

if __name__ == "__main__":
    obj = PowerReviews()
    obj.get_info()