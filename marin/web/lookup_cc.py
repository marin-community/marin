import io

import requests
import json
import warcio
from urllib.parse import quote_plus

# Please note: f-strings require Python 3.6+

# The URL of the Common Crawl Index server
# CC_INDEX_SERVER = 'http://index.commoncrawl.org/'
CC_INDEX_SERVER = 'http://localhost:8080/'

# The Common Crawl index you want to query
INDEX_NAME = '2023-40'      # Replace with the latest index name

# Function to search the Common Crawl Index
def search_cc_index(url, index_name=INDEX_NAME):
    encoded_url = quote_plus(url)
    index_url = f'{CC_INDEX_SERVER}CC-MAIN-{index_name}-index?url={encoded_url}&output=json'
    response = requests.get(index_url)
    # print("Response from CCI:", response.text)  # Output the response from the server
    if response.status_code == 200:
        records = response.text.strip().split('\n')
        return [json.loads(record) for record in records]
    else:
        return None

# Function to fetch the content from Common Crawl
def fetch_page_from_cc(records):
    result = bytearray()
    for record in records:
        offset, length = int(record['offset']), int(record['length'])
        s3_url = f'https://data.commoncrawl.org/{record["filename"]}'
        response = requests.get(s3_url, headers={'Range': f'bytes={offset}-{offset+length-1}'})
        if response.status_code == 206:
            with io.BytesIO(response.content) as stream:
                i = 0
                for record in warcio.ArchiveIterator(stream):
                    html = record.content_stream().read()
                    i += 1
                    if i > 1:
                        print("More than one record found")
                        break
                    result += html
        else:
            print(f"Failed to fetch data: {response.status_code}")

    if result:
        import chardet
        enc = chardet.detect(result)
        if enc['encoding'] is not None:
            try:
                return result.decode(enc['encoding'])
            except UnicodeDecodeError:
                pass

        try:
            return result.decode('utf-8')
        except UnicodeDecodeError:
            pass

    return None


if __name__ == '__main__':
    import time
    time_in = time.time()
    for i in range(20):
        # Search the index for the target URL
        # The URL you want to look up in the Common Crawl index
        target_url = 'http://canalciencia.us.es/abierto-el-plazo-para-participar-en-la-noche-europea-de-los-investigadores-2017/'

        records = search_cc_index(target_url)
        if records:
            print(f"Found {len(records)} records for {target_url}")

            # Fetch the page content from the first record
            content = fetch_page_from_cc(records)
            # print(content)
        else:
            print(f"No records found for {target_url}")

    print("Time taken:", time.time() - time_in)