import requests
from bs4 import BeautifulSoup

url = "https://www.sharesansar.com/company/NICA"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

company_info = soup.find("div", class_="company-detail")
if company_info:
    print(company_info.text[:500])
else:
    print("Company detail not found")
