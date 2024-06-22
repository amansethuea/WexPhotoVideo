import re
import os
import sys
import csv
import json
import requests
from datetime import datetime

class TrustPilot(object):
    def __init__(self):
        self.api_key = "OSGJRidu1dtDuDBaiq16IU63Rqn4ZSX1"
        # Date range can be 12 months, 6 months, 3 months, 30 days, 7 days, 2 days, last day, yesterday, today, this month
        #date_range = "12 months"
        self.date_range = "2024-06-21"

        if sys.platform.startswith("win"):
            self.trust_pilot_data_path = os.getcwd() + "/../Data_Files/trustpilot_all_reviews_api.csv"
        elif sys.platform.startswith("darwin"):
            self.trust_pilot_data_path = os.getcwd() + "/Data_Files/trustpilot_all_reviews_api.csv"
        elif sys.platform.startswith("linux"):
            self.trust_pilot_data_path = os.getcwd() + "/../Data_Files/trustpilot_all_reviews_api.csv"
        
    # Fetching business unit ID
    def get_business_id(self):
        business_unit_id_url = f"https://api.trustpilot.com/v1/business-units/find/?apikey={self.api_key}&name=www.wexphotovideo.com"
        get_b_id_data = requests.get(business_unit_id_url)
        status, res = get_b_id_data.status_code, get_b_id_data.content
        if status == 200:
            data = json.loads(res)
            b_id = data["id"]
            return b_id
        else:
            print(f"ERROR: Status {status}. Please check.")
            return False
    
    def get_time_range(self):
        if re.search(". *[a-zA-Z]. *", self.date_range):
            get_date_range = self.date_range.upper()
            if get_date_range in ["12 MONTHS", "LAST 12 MONTHS"]:
                calculate_date = datetime.date.today() - datetime.timedelta(365)
                year, month, day = calculate_date.year, calculate_date.month, calculate_date.day
                start_date = str(year)+"-"+str(month)+"-"+str(day)+"T00:00:00"
                return start_date
            elif get_date_range in ["6 MONTHS", "LAST 6 MONTHS"]:
                calculate_date = datetime.date.today() - datetime.timedelta(180)
                year, month, day = calculate_date.year, calculate_date.month, calculate_date.day
                start_date = str(year)+"-"+str(month)+"-"+str(day)+"T00:00:00"
                return start_date
            elif get_date_range in ["3 MONTHS", "LAST 3 MONTHS"]:
                calculate_date = datetime.date.today() - datetime.timedelta(90)
                year, month, day = calculate_date.year, calculate_date.month, calculate_date.day
                start_date = str(year)+"-"+str(month)+"-"+str(day)+"T00:00:00"
                return start_date
            elif get_date_range in ["30 DAYS", "LAST 30 DAYS"]:
                calculate_date = datetime.date.today() - datetime.timedelta(60)
                year, month, day = calculate_date.year, calculate_date.month, calculate_date.day
                start_date = str(year)+"-"+str(month)+"-"+str(day)+"T00:00:00"
                return start_date
            elif get_date_range in ["7 DAYS", "LAST 7 DAYS"]:
                calculate_date = datetime.date.today() - datetime.timedelta(7)
                year, month, day = calculate_date.year, calculate_date.month, calculate_date.day
                start_date = str(year)+"-"+str(month)+"-"+str(day)+"T00:00:00"
                return start_date
            elif get_date_range in ["2 DAYS", "LAST 2 DAYS"]:
                calculate_date = datetime.date.today() - datetime.timedelta(2)
                year, month, day = calculate_date.year, calculate_date.month, calculate_date.day
                start_date = str(year)+"-"+str(month)+"-"+str(day)+"T00:00:00"
                return start_date
            elif get_date_range in ["1 DAY", "ONE DAY", "LAST DAY", "YESTERDAY"]:
                calculate_date = datetime.date.today() - datetime.timedelta(1)
                year, month, day = calculate_date.year, calculate_date.month, calculate_date.day
                start_date = str(year)+"-"+str(month)+"-"+str(day)+"T00:00:00"
                return start_date
            elif get_date_range in ["TODAY"]:
                calculate_date = datetime.date.today()
                year, month, day = calculate_date.year, calculate_date.month, calculate_date.day
                start_date = str(year)+"-"+str(month)+"-"+str(day)+"T00:00:00"
                return start_date
            elif get_date_range in ["THIS MONTH"]:
                input_dt = datetime.date.today()
                res = input_dt.replace(day=1)
                find_days_diff = input_dt - res
                find_days_diff = str(find_days_diff)
                get_day_no_str = find_days_diff.split(",")[0]
                get_day_no = get_day_no_str.split(" ")[0]

                calculate_date = datetime.date.today() - datetime.timedelta(int(get_day_no))
                year, month, day = calculate_date.year, calculate_date.month, calculate_date.day
                start_date = str(year)+"-"+str(month)+"-"+str(day)+"T00:00:00"
                return start_date
            else:
                print("Invalid date. Please choose from the date range selection only")
                return False
        else:
            return self.date_range+"T:00:00:00"

    # Create a list of URLs as per page_no_range global variable defined above
    def create_url(self, stars="all", page_no_range=100):
        review_url_list = []
        business_unit_id = self.get_business_id()
        start_date = self.get_time_range()

        for i in range(1, page_no_range + 1):
            if stars == "all":
                reviews_url = f"https://api.trustpilot.com/v1/business-units/{business_unit_id}/reviews?apikey={self.api_key}&page={i}&language=en&startDateTime={start_date}"
            else:
                reviews_url = f"https://api.trustpilot.com/v1/business-units/{business_unit_id}/reviews?apikey={self.api_key}&stars={stars}&page={i}&language=en&startDateTime={start_date}"
            review_url_list.append(reviews_url)
        
        return review_url_list

    # Get all reviews data
    def get_info(self, stars="all", page_no_range=100):
        final_info_dict_list = []
        get_url_list = self.create_url(stars, page_no_range)
        for i in (get_url_list):
            review_data = requests.get(i)
            status, res = review_data.status_code, review_data.content
            if status == 200:
                parsed_dict = json.loads(res)
                for page_dict in parsed_dict['reviews']:
                    final_info_dict_list.append({'review_title': page_dict['title'],
                                                'review_content': page_dict['text'],
                                                'review_experience_date': page_dict['experiencedAt'],
                                                'review_created_date': page_dict['createdAt'],
                                                'review_stars': page_dict['stars'],
                                                'reviewer_name': page_dict['consumer']['displayName'],
                                                'review_location': page_dict['consumer']['displayLocation'],
                                                'reviews_given': page_dict['consumer']['numberOfReviews']
                                                })
            else:
                print(f"ERROR: Status {status}. Unable to fetch data for page: {i+1}.")
                return False

        return final_info_dict_list

    def clear_csv(self):
        fo = open(self.trust_pilot_data_path, "w")
        fo.writelines("")
        fo.close()
    
    def write_csv_data(self, stars="all", page_no_range=100):
        # Get the page dict data
        page_data_dict_list = self.get_info(stars, page_no_range)
        file_name = self.trust_pilot_data_path
        for page_data in page_data_dict_list:
            try:
                with open(file_name, 'a', encoding = 'utf8', newline = '') as csvfile:
                    fieldnames = ['Title', 'Content', 'Date of Experience',
                                    'Review Date', 'Rating', 'Reviewer Name', 'Location',
                                    'No of reviews given']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    is_file_empty = os.stat(file_name).st_size
                    if is_file_empty == 0:
                        writer.writeheader()
                    writer.writerow({'Title': page_data['review_title'],
                                    'Content': page_data['review_content'],
                                    'Date of Experience': page_data['review_experience_date'],
                                    'Review Date': page_data['review_created_date'],
                                    'Rating': page_data['review_stars'],
                                    'Reviewer Name': page_data['reviewer_name'],
                                    'Location': page_data['review_location'],
                                    'No of reviews given': page_data['reviews_given']})
                    csvfile.close()
            except:
                print(f"Something unexpected happened. Please check the output.")
        print("Trust Pilot Reviews fetched successfully. Please check trustpilot_all_reviews_api.csv")

if __name__ == "__main__":
    obj = TrustPilot()
    obj.clear_csv()
    obj.write_csv_data()
