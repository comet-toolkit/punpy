#/bin/bash
pdfFile=$1
echo "Processing $pdfFile"
numberOfPages=$(/usr/local/bin/identify "$pdfFile" 2>/dev/null | wc -l | tr -d ' ')
#Identify gets info for each page, dump stderr to dev null
#count the lines of output
#trim the whitespace from the wc -l outout
echo "The number of pages is: $numberOfPages"