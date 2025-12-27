mkdir -p SEVIR
cd SEVIR

# Download the SEVIR VIL dataset
aws s3 cp --no-sign-request s3://sevir/CATALOG.csv CATALOG.csv
aws s3 sync --no-sign-request s3://sevir/data/vil ./vil

# If the file is too large, you can try downloading it year by year.
# aws s3 sync --no-sign-request s3://sevir/data/vil/2017 ./vil/2017
# aws s3 sync --no-sign-request s3://sevir/data/vil/2018 ./vil/2018
# aws s3 sync --no-sign-request s3://sevir/data/vil/2019 ./vil/2019

echo "Done!"