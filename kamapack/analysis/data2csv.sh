# Gzip all *.data to *.csv data 
# substitute " " with "," sep

for file in DATA/*.data
do
	sed -i 's/ /,/g' $file
    mv -- $file ${file%.data}.csv
	gzip ${file%.data}.csv
done
