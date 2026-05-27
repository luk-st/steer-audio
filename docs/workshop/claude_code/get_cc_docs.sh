
curl -s https://docs.anthropic.com/sitemap.xml | grep -o 'https://[^<]*/en/docs/claude-code[^<]*' | sort -u > urls.txt

while read url; do
    filename=$(basename "$url")
    echo ${url}
    curl -o "${filename}.md" "$url"
done < urls.txt
