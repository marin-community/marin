import json
import logging

import apache_beam as beam
import fsspec
from apache_beam.options.pipeline_options import PipelineOptions
from warcio.archiveiterator import ArchiveIterator


from process_rpv2 import extract_from_html

logger = logging.getLogger(__name__)


# Placeholder for the filter function
def filter_urls(element):
    # Apply your filter logic here
    # Return True if the element passes the filter, False otherwise
    pass


class ProcessHtmlContentFn(beam.DoFn):
    def process(self, element, *args, **kwargs):
        segment_file, urls_info = element
        urls_set = set([url_info["url"] for url_info in urls_info])  # Convert URLs to a set for efficient lookup

        # Placeholder for fetching the WARC file from S3 or another location
        # Here, we'll use a simple HTTP request as an example, but in practice,
        # you would fetch this from your storage solution (e.g., S3, GCS)
        # warc_file_url = f'https://example.com/path/to/warc/files/{segment_file}'
        try:
            with fsspec.open("s3://commoncrawl/" + segment_file, "rb") as f:
                for record in ArchiveIterator(f):
                    if record.rec_type == "response":
                        # Extract the target URI from the WARC record
                        target_uri = record.rec_headers.get_header("WARC-Target-URI")
                        if target_uri in urls_set:
                            # Process the HTML content of the record here
                            # This is where you'd integrate Readability.js and html2text
                            html_content = record.content_stream().read()
                            # Placeholder for processing the HTML content
                            processed_content = extract_from_html(target_uri, html_content)

                            # Emit the processed content for further use or storage
                            yield target_uri, processed_content
                            break
        except Exception as e:
            logger.exception(f"Error processing segment file {segment_file}: {e}")


class GroupBySegmentFn(beam.DoFn):
    def process(self, element):
        # Extract the segment file and URL from the element
        segment_file, url_info = element
        # Yield a tuple of the segment file and the URL info
        yield (segment_file, url_info)


def run():
    pipeline_options = PipelineOptions()
    with beam.Pipeline(options=pipeline_options) as p:
        # Read the RPV2 metadata from GCS
        urls = (
            p
            | "ReadRPV2Metadata" >> beam.io.ReadFromText("gs://your_bucket/your_rpv2_metadata_files*")
            | "ParseJSON" >> beam.Map(lambda x: json.loads(x))
        )

        # Apply your filtering criteria
        filtered_urls = (
            urls
            | "FilterURLs" >> beam.Filter(filter_urls)
            | "MapToSegmentFile" >> beam.Map(lambda x: (x["cc_segment"], x))
        )

        # Group URLs by their corresponding Common Crawl segment
        grouped_by_segment = filtered_urls | "GroupBySegmentFile" >> beam.GroupByKey()

        # Process the HTML content for each segment
        processed_html = grouped_by_segment | "ProcessHTMLContent" >> beam.ParDo(ProcessHtmlContentFn())

        # You can then write the processed HTML (now in markdown) to GCS or any other sink
        # processed_html | 'WriteToGCS' >> beam.io.WriteToText('gs://your_output_bucket/output')


if __name__ == "__main__":
    run()
