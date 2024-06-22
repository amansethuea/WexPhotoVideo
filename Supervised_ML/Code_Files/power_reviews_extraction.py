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
        self.date_range = "21/06/2024"

        if sys.platform.startswith("win"):
            self.power_reviews_data_path = os.getcwd() + "/../Data_Files/power_reviews_all_api.csv"
        elif sys.platform.startswith("darwin"):
            self.power_reviews_data_path = os.getcwd() + "/Data_Files/power_reviews_all_api.csv"
    
    def get_time_range(self):
        if re.search(". *[a-zA-Z]. *", self.date_range):
            get_date_range = self.date_range.upper()
            if get_date_range in ["12 MONTHS", "LAST 12 MONTHS"]:
                calculate_date = datetime.date.today() - datetime.timedelta(365)
                unix_time= calculate_date.strftime("%s")
                # return str(int(unix_time) * 1000)
                return unix_time
            elif get_date_range in ["6 MONTHS", "LAST 6 MONTHS"]:
                calculate_date = datetime.date.today() - datetime.timedelta(180)
                unix_time= calculate_date.strftime("%s")
                return unix_time
            elif get_date_range in ["3 MONTHS", "LAST 3 MONTHS"]:
                calculate_date = datetime.date.today() - datetime.timedelta(90)
                unix_time= calculate_date.strftime("%s")
                return unix_time
            elif get_date_range in ["30 DAYS", "LAST 30 DAYS"]:
                calculate_date = datetime.date.today() - datetime.timedelta(60)
                unix_time= calculate_date.strftime("%s")
                return unix_time
            elif get_date_range in ["7 DAYS", "LAST 7 DAYS"]:
                calculate_date = datetime.date.today() - datetime.timedelta(7)
                unix_time= calculate_date.strftime("%s")
                return unix_time
            elif get_date_range in ["2 DAYS", "LAST 2 DAYS"]:
                calculate_date = datetime.date.today() - datetime.timedelta(2)
                unix_time= calculate_date.strftime("%s")
                return unix_time
            elif get_date_range in ["1 DAY", "ONE DAY", "LAST DAY"]:
                calculate_date = datetime.date.today() - datetime.timedelta(1)
                unix_time= calculate_date.strftime("%s")
                return unix_time
            elif get_date_range in ["TODAY"]:
                calculate_date = datetime.date.today()
                unix_time= calculate_date.strftime("%s")
                return unix_time
            elif get_date_range in ["THIS MONTH"]:
                input_dt = datetime.date.today()
                res = input_dt.replace(day=1)
                find_days_diff = input_dt - res
                find_days_diff = str(find_days_diff)
                get_day_no_str = find_days_diff.split(",")[0]
                get_day_no = get_day_no_str.split(" ")[0]

                calculate_date = datetime.date.today() - datetime.timedelta(int(get_day_no))
                unix_time= calculate_date.strftime("%s")
                return unix_time
            else:
                print("Invalid date. Please choose from the date range selection only")
                return False
        else:
            unix_time = time.mktime(datetime.datetime.strptime(self.date_range, "%d/%m/%Y").timetuple())
            # return str(int(unix_time) * 1000)
            return str(int(unix_time))

    def url_formation(self):
        get_time = self.get_time_range()
        # Fetch the first page initially
        #review_data_url = f"https://readservices-b2c.powerreviews.com/m/{merchant_id}/reviews?apikey={api_key}&paging.from=1&paging.size=25&date={get_time}&sort=Newest"
        review_data_url = f"https://readservices-b2c.powerreviews.com/m/{self.merchant_id}/reviews?apikey={self.api_key}&paging.from=1&paging.size=25&date={get_time}&filters=&search=&sort=Newest&image_only=false&page_locale=en_GB"
        review_data = requests.get(review_data_url)
        data_bytes = review_data.content
        my_json_dict = json.loads(data_bytes)
        # Fetch total no. of pages
        fetch_total_pages = int(my_json_dict['paging']['pages_total'])
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
        fo = open(self.power_reviews_data_path, "w")
        fo.writelines("")
        fo.close()

    def write_csv_data(self, file_name, review_created_date, page_id, review_rating, review_headline, comment, review_location, reviewer_nickname):
        file_name = self.power_reviews_data_path
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
        file_name = self.power_reviews_data_path
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