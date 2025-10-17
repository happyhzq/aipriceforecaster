"""
MySQL数据适配器 - 从MySQL数据库读取市场数据
"""
import os
import pandas as pd
import pymysql
from sqlalchemy import create_engine
from typing import Optional, Dict, Any
from ..utils.logger import get_logger
from sqlalchemy import text


class MySQLAdapter:
    """MySQL数据源适配器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化MySQL连接
        
        Args:
            config: 数据库配置字典，包含:
                - host: 数据库主机
                - port: 端口号
                - user: 用户名
                - password: 密码
                - database: 数据库名
                - charset: 字符集(默认utf8mb4)
        """
        self.logger = get_logger("mysql_adapter")
        self.config = config
        self.engine = None
        self._init_connection()
    
    def _init_connection(self):
        """初始化数据库连接"""
        try:
            # 构建连接字符串
            host = self.config.get('host', 'localhost')
            port = self.config.get('port', 3306)
            user = self.config.get('user', 'root')
            password = self.config.get('password', '')
            database = self.config.get('database', '')
            charset = self.config.get('charset', 'utf8mb4')
            
            # 优先从环境变量读取敏感信息
            password = os.environ.get('MYSQL_PASSWORD', password)
            
            # 创建SQLAlchemy引擎
            connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset={charset}"
            self.engine = create_engine(connection_string, pool_pre_ping=True)
            
            # 测试连接
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                self.logger.info(f"MySQL connection established: {host}:{port}/{database}")
                
        except Exception as e:
            self.logger.error(f"Failed to connect to MySQL: {str(e)}")
            raise
    
    def fetch_kline_data(self, 
                        table_name: str,
                        ticker: Optional[str] = None,
                        freq: Optional[int] = None,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None,
                        columns_mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        从MySQL获取K线数据
        
        Args:
            table_name: 表名
            ticker: 标的代码（可选）
            freq: 数据频率（可选）
            start_date: 开始日期 (格式: 'YYYY-MM-DD HH:MM:SS')
            end_date: 结束日期
            columns_mapping: 列名映射，如 {'dt': 'timestamp', 'symbol': 'ticker'}
        
        Returns:
            pd.DataFrame: 包含K线数据的DataFrame
        """
        try:
            # 构建查询SQL
            query = f"SELECT * FROM {table_name} WHERE 1=1"
            params = {}
            
            if ticker:
                query += " AND param_symbol = %(ticker)s"
                params['ticker'] = ticker
            
            if freq:
                query += " AND param_period = %(freq)s"
                params['freq'] = freq
            
            if start_date:
                query += " AND trade_datetime >= %(start_date)s"
                params['start_date'] = start_date
            
            if end_date:
                query += " AND trade_datetime <= %(end_date)s"
                params['end_date'] = end_date
            
            query += " ORDER BY trade_datetime ASC"

            # 执行查询
            self.logger.info(f"Executing query: {query[:100]}...")
            df = pd.read_sql_query(query, self.engine, params=params)
            
            # 列名映射
            if columns_mapping:
                df = df.rename(columns=columns_mapping)
            
            # 确保必要的列存在
            required_cols = ['timestamp', 'ticker', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                self.logger.warning(f"Missing columns: {missing_cols}")
            
            # 转换时间戳列
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            self.logger.info(f"Fetched {len(df)} rows from {table_name}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching data from MySQL: {str(e)}")
            raise
    
    def fetch_multiple_tickers(self,
                              table_name: str,
                              tickers: list,
                              freq:Optional[list] = None,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None,
                              columns_mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        获取多个标的的数据
        
        Args:
            table_name: 表名
            tickers: 标的代码列表
            freq: 频率
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            pd.DataFrame: 合并后的数据
        """
        all_data = []
        
        for ticker in tickers:
            self.logger.info(f"Fetching data for {ticker}")
            df = self.fetch_kline_data(
                table_name=table_name,
                ticker=ticker,
                freq = freq,
                start_date=start_date,
                end_date=end_date,
                columns_mapping=columns_mapping
            )
            all_data.append(df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.sort_values(['ticker', 'timestamp'])
            return combined_df
        else:
            return pd.DataFrame()
    
    def execute_custom_query(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """
        执行自定义SQL查询
        
        Args:
            query: SQL查询语句
            params: 查询参数
        
        Returns:
            pd.DataFrame: 查询结果
        """
        try:
            df = pd.read_sql_query(query, self.engine, params=params)
            return df
        except Exception as e:
            self.logger.error(f"Error executing custom query: {str(e)}")
            raise
    
    def close(self):
        """关闭数据库连接"""
        if self.engine:
            self.engine.dispose()
            self.logger.info("MySQL connection closed")


def create_sample_table_schema():
    """
    返回建议的MySQL表结构SQL
    """
    return """
    -- 建议的K线数据表结构
    CREATE TABLE IF NOT EXISTS kline_data (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        timestamp DATETIME NOT NULL,
        ticker VARCHAR(20) NOT NULL,
        open DECIMAL(20, 4) NOT NULL,
        high DECIMAL(20, 4) NOT NULL,
        low DECIMAL(20, 4) NOT NULL,
        close DECIMAL(20, 4) NOT NULL,
        volume BIGINT NOT NULL,
        hold BIGINT DEFAULT 0,
        turnover DECIMAL(20, 2) DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        
        INDEX idx_timestamp (timestamp),
        INDEX idx_ticker (ticker),
        INDEX idx_ticker_timestamp (ticker, timestamp),
        UNIQUE KEY unique_ticker_timestamp (ticker, timestamp)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    
    -- 可选：分区表（按月分区）
    ALTER TABLE kline_data PARTITION BY RANGE (YEAR(timestamp) * 100 + MONTH(timestamp)) (
        PARTITION p202501 VALUES LESS THAN (202502),
        PARTITION p202502 VALUES LESS THAN (202503),
        PARTITION p202503 VALUES LESS THAN (202504),
        -- 添加更多分区...
        PARTITION p_future VALUES LESS THAN MAXVALUE
    );
    """