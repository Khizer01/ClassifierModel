import aiosqlite
from typing import List, Dict, Optional
from config import get_settings
import os

settings = get_settings()

class DatabaseManager:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or settings.database_path

    async def _ensure_schema(self, db: aiosqlite.Connection) -> None:
        await db.execute("PRAGMA foreign_keys = ON")
        await db.executescript(
            """
            CREATE TABLE IF NOT EXISTS keywords (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keyword TEXT NOT NULL UNIQUE
            );

            CREATE TABLE IF NOT EXISTS meta_ads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keyword_id INTEGER,
                body_text TEXT,
                page_name TEXT,
                title TEXT,
                description TEXT,
                FOREIGN KEY(keyword_id) REFERENCES keywords(id)
            );

            CREATE INDEX IF NOT EXISTS idx_meta_ads_keyword_id ON meta_ads(keyword_id);
            """
        )
        await db.commit()

    async def ensure_initialized(self) -> None:
        if not self.db_path:
            return
        try:
            if os.path.exists(self.db_path) and os.path.getsize(self.db_path) == 0:
                pass
        except OSError:
            pass

        async with aiosqlite.connect(self.db_path) as db:
            await self._ensure_schema(db)
    
    async def get_ads_by_keyword(self, keyword: str) -> List[Dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await self._ensure_schema(db)

            try:
                cur = await db.execute("SELECT COUNT(*) FROM meta_ads")
                meta_count = (await cur.fetchone())[0]
                await cur.close()
                cur = await db.execute("SELECT COUNT(*) FROM keywords")
                kw_count = (await cur.fetchone())[0]
                await cur.close()
                if meta_count == 0 or kw_count == 0:
                    print(
                        f"[WARNING] Database '{self.db_path}' has no data yet (meta_ads={meta_count}, keywords={kw_count})."
                    )
            except Exception:
                pass
            
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
            await self._ensure_schema(db)
            
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
            await self._ensure_schema(db)
            
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
            await self._ensure_schema(db)
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
