# dumbly just processes all the stackexchange data (that end in stackexchange.com.7z)

D=$(dirname $0)

GCS_PREFIX="gs://levanter-data/markweb-raw/stackexchange/"

# line looks like https://archive.org/download/stackexchange/3dprinting.stackexchange.com.7z
# we want "gs://levanter-data/markweb-raw/stackexchange/archive.org/stackexchange/3dprinting.stackexchange.com.7z"
# (That is, replace https:// with gs://levanter-data/markweb-raw/stackexchange/

for path in $(grep stackexchange.com ${D}/stack_exchange_urls.tsv); do
    echo "Processing $path"
    gcspath=$(echo $path | sed 's|https://|gs://levanter-data/markweb-raw/stackexchange/|')
    echo $gcspath
    python ${D}/process_stack_exchange.py $gcspath
done