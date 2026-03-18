from enum import Enum


class DatasourceType(Enum):
    FILE = "upload_file"
    NOTION = "notion_import"
    WEBSITE = "website_crawl"
    
    def __str__(self):
        return self.value
