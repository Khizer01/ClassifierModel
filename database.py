import aiosqlite
from typing import List, Dict, Optional
from config import get_settings

settings = get_settings()

class DatabaseManager:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or settings.database_path
    
    async def get_ads_by_keyword(self, keyword: str) -> List[Dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            query = """
                SELECT 
                    m.id,
                    m.body_text,
                    m.page_name,
                    m.title,
                    m.description,
                    k.keyword
                FROM meta_ads m
                JOIN keywords k ON m.keyword_id = k.id
                WHERE LOWER(k.keyword) = LOWER(?)
                AND m.body_text IS NOT NULL
                AND m.body_text != ''
            """
            
            async with db.execute(query, (keyword,)) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
    
    async def get_ad_by_id(self, ad_id: int) -> List[Dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            query = """
                SELECT 
                    m.id,
                    m.body_text,
                    m.page_name,
                    m.title,
                    m.description,
                    k.keyword
                FROM meta_ads m
                JOIN keywords k ON m.keyword_id = k.id
                WHERE m.id = ?
                AND m.body_text IS NOT NULL
                AND m.body_text != ''
            """
            
            async with db.execute(query, (ad_id,)) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
    
    async def get_all_ads(self, limit: Optional[int] = None) -> List[Dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            query = """
                SELECT 
                    m.id,
                    m.body_text,
                    m.page_name,
                    m.title,
                    m.description,
                    k.keyword
                FROM meta_ads m
                JOIN keywords k ON m.keyword_id = k.id
                WHERE m.body_text IS NOT NULL
                AND m.body_text != ''
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            async with db.execute(query) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
    
    async def get_ad_count_by_keyword(self, keyword: str) -> int:
        async with aiosqlite.connect(self.db_path) as db:
            query = """
                SELECT COUNT(*) as count
                FROM meta_ads m
                JOIN keywords k ON m.keyword_id = k.id
                WHERE LOWER(k.keyword) = LOWER(?)
                AND m.body_text IS NOT NULL
                AND m.body_text != ''
            """
            
            async with db.execute(query, (keyword,)) as cursor:
                row = await cursor.fetchone()
                return row[0] if row else 0
