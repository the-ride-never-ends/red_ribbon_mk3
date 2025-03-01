"""
SQL Repository for VariableCodebook
"""
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import sqlite3  # Using SQLite for demo, but could be any SQL database
from .codebook_models import Codebook, CodebookGroup, Variable, CategoryValue, DataType

logger = logging.getLogger(__name__)

class SQLRepository:
    """
    Repository for accessing variable codebook data from SQL database
    """
    
    def __init__(self, connection_string: str = "memory"):
        """
        Initialize the SQL repository
        
        Args:
            connection_string: Database connection string
        """
        logger.info("SQLRepository initializing")
        self.connection_string = connection_string
        
        # For demo purposes, using SQLite in-memory database
        if connection_string == "memory":
            self.conn = sqlite3.connect(":memory:")
        else:
            self.conn = sqlite3.connect(connection_string)
        
        self._setup_demo_db()
        logger.info("SQLRepository initialized")
    
    def _setup_demo_db(self):
        """Set up demo database schema and seed data"""
        cursor = self.conn.cursor()
        
        # Create tables
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS codebooks (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            version TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS codebook_groups (
            id TEXT PRIMARY KEY,
            codebook_id TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            parent_group_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (codebook_id) REFERENCES codebooks (id),
            FOREIGN KEY (parent_group_id) REFERENCES codebook_groups (id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS variables (
            id TEXT PRIMARY KEY,
            codebook_id TEXT NOT NULL,
            group_id TEXT,
            name TEXT NOT NULL,
            description TEXT,
            data_type TEXT NOT NULL,
            required INTEGER NOT NULL,
            min_value REAL,
            max_value REAL,
            default_value TEXT,
            format_pattern TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (codebook_id) REFERENCES codebooks (id),
            FOREIGN KEY (group_id) REFERENCES codebook_groups (id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS category_values (
            id TEXT PRIMARY KEY,
            variable_id TEXT NOT NULL,
            code TEXT NOT NULL,
            label TEXT NOT NULL,
            description TEXT,
            order_num INTEGER,
            is_default INTEGER NOT NULL,
            FOREIGN KEY (variable_id) REFERENCES variables (id)
        )
        ''')
        
        # Insert demo data
        timestamp = datetime.now().isoformat()
        
        # Insert a demo codebook
        cursor.execute('''
        INSERT OR IGNORE INTO codebooks (id, name, description, version, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', ('cb1', 'Survey Variables', 'Codebook for survey variables', '1.0', timestamp, timestamp))
        
        # Insert demo groups
        cursor.execute('''
        INSERT OR IGNORE INTO codebook_groups (id, codebook_id, name, description, parent_group_id, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', ('g1', 'cb1', 'Demographics', 'Demographic variables', None, timestamp, timestamp))
        
        cursor.execute('''
        INSERT OR IGNORE INTO codebook_groups (id, codebook_id, name, description, parent_group_id, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', ('g2', 'cb1', 'Survey Responses', 'Survey response variables', None, timestamp, timestamp))
        
        # Insert demo variables
        cursor.execute('''
        INSERT OR IGNORE INTO variables 
        (id, codebook_id, group_id, name, description, data_type, required, min_value, max_value, default_value, format_pattern, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', ('v1', 'cb1', 'g1', 'age', 'Age in years', 'integer', 1, 0, 120, None, None, timestamp, timestamp))
        
        cursor.execute('''
        INSERT OR IGNORE INTO variables 
        (id, codebook_id, group_id, name, description, data_type, required, min_value, max_value, default_value, format_pattern, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', ('v2', 'cb1', 'g1', 'gender', 'Gender identity', 'categorical', 1, None, None, None, None, timestamp, timestamp))
        
        cursor.execute('''
        INSERT OR IGNORE INTO variables 
        (id, codebook_id, group_id, name, description, data_type, required, min_value, max_value, default_value, format_pattern, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', ('v3', 'cb1', 'g2', 'satisfaction', 'Satisfaction score', 'integer', 1, 1, 5, None, None, timestamp, timestamp))
        
        # Insert category values for gender
        cursor.execute('''
        INSERT OR IGNORE INTO category_values (id, variable_id, code, label, description, order_num, is_default)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', ('cv1', 'v2', 'M', 'Male', 'Male gender identity', 1, 0))
        
        cursor.execute('''
        INSERT OR IGNORE INTO category_values (id, variable_id, code, label, description, order_num, is_default)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', ('cv2', 'v2', 'F', 'Female', 'Female gender identity', 2, 0))
        
        cursor.execute('''
        INSERT OR IGNORE INTO category_values (id, variable_id, code, label, description, order_num, is_default)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', ('cv3', 'v2', 'NB', 'Non-binary', 'Non-binary gender identity', 3, 0))
        
        cursor.execute('''
        INSERT OR IGNORE INTO category_values (id, variable_id, code, label, description, order_num, is_default)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', ('cv4', 'v2', 'O', 'Other', 'Other gender identity', 4, 0))
        
        self.conn.commit()
    
    def get_codebook(self, codebook_id: str) -> Optional[Codebook]:
        """
        Get a complete codebook by ID
        
        Args:
            codebook_id: ID of the codebook to retrieve
            
        Returns:
            Codebook object or None if not found
        """
        logger.info(f"Getting codebook with ID: {codebook_id}")
        
        try:
            # Get codebook basic info
            cursor = self.conn.cursor()
            cursor.execute("SELECT id, name, description, version, created_at, updated_at FROM codebooks WHERE id = ?", (codebook_id,))
            row = cursor.fetchone()
            
            if not row:
                logger.warning(f"Codebook with ID {codebook_id} not found")
                return None
                
            codebook = Codebook(
                id=row[0],
                name=row[1],
                description=row[2],
                version=row[3],
                created_at=datetime.fromisoformat(row[4]),
                updated_at=datetime.fromisoformat(row[5]),
                groups=[],
                variables=[]
            )
            
            # Get groups
            cursor.execute("SELECT id, name, description, parent_group_id, created_at, updated_at FROM codebook_groups WHERE codebook_id = ?", (codebook_id,))
            group_rows = cursor.fetchall()
            
            groups_dict = {}  # For easy lookup when building the hierarchy
            for row in group_rows:
                group = CodebookGroup(
                    id=row[0],
                    name=row[1],
                    description=row[2],
                    parent_group_id=row[3],
                    created_at=datetime.fromisoformat(row[4]),
                    updated_at=datetime.fromisoformat(row[5]),
                    variables=[]
                )
                groups_dict[group.id] = group
                codebook.groups.append(group)
            
            # Get variables
            cursor.execute("SELECT id, name, description, data_type, group_id, required, min_value, max_value, default_value, format_pattern, created_at, updated_at FROM variables WHERE codebook_id = ?", (codebook_id,))
            variable_rows = cursor.fetchall()
            
            for row in variable_rows:
                variable = Variable(
                    id=row[0],
                    name=row[1],
                    description=row[2],
                    data_type=DataType(row[3]),
                    required=bool(row[5]),
                    min_value=row[6],
                    max_value=row[7],
                    default_value=row[8],
                    format_pattern=row[9],
                    created_at=datetime.fromisoformat(row[10]),
                    updated_at=datetime.fromisoformat(row[11]),
                    categories=[]
                )
                
                # Add to both codebook and relevant group
                codebook.variables.append(variable)
                
                group_id = row[4]
                if group_id and group_id in groups_dict:
                    groups_dict[group_id].variables.append(variable)
                
                # Get category values if this is a categorical variable
                if variable.data_type == DataType.CATEGORICAL:
                    cursor.execute("SELECT code, label, description, order_num, is_default FROM category_values WHERE variable_id = ?", (variable.id,))
                    category_rows = cursor.fetchall()
                    
                    variable.categories = []
                    for cat_row in category_rows:
                        category = CategoryValue(
                            code=cat_row[0],
                            label=cat_row[1],
                            description=cat_row[2],
                            order=cat_row[3],
                            is_default=bool(cat_row[4])
                        )
                        variable.categories.append(category)
            
            logger.info(f"Retrieved codebook {codebook.name} with {len(codebook.variables)} variables and {len(codebook.groups)} groups")
            return codebook
            
        except Exception as e:
            logger.error(f"Error retrieving codebook: {str(e)}")
            return None
    
    def get_all_codebooks(self) -> List[Codebook]:
        """
        Get all codebooks (without full details)
        
        Returns:
            List of codebook summary objects
        """
        logger.info("Getting all codebooks")
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT id, name, description, version, created_at, updated_at FROM codebooks")
            rows = cursor.fetchall()
            
            codebooks = []
            for row in rows:
                codebook = Codebook(
                    id=row[0],
                    name=row[1],
                    description=row[2],
                    version=row[3],
                    created_at=datetime.fromisoformat(row[4]),
                    updated_at=datetime.fromisoformat(row[5])
                )
                codebooks.append(codebook)
            
            logger.info(f"Retrieved {len(codebooks)} codebooks")
            return codebooks
            
        except Exception as e:
            logger.error(f"Error retrieving all codebooks: {str(e)}")
            return []
    
    def get_variable(self, variable_id: str) -> Optional[Variable]:
        """
        Get a variable by ID
        
        Args:
            variable_id: ID of the variable to retrieve
            
        Returns:
            Variable object or None if not found
        """
        logger.info(f"Getting variable with ID: {variable_id}")
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT id, name, description, data_type, required, min_value, max_value, 
                       default_value, format_pattern, created_at, updated_at 
                FROM variables WHERE id = ?
            """, (variable_id,))
            row = cursor.fetchone()
            
            if not row:
                logger.warning(f"Variable with ID {variable_id} not found")
                return None
                
            variable = Variable(
                id=row[0],
                name=row[1],
                description=row[2],
                data_type=DataType(row[3]),
                required=bool(row[4]),
                min_value=row[5],
                max_value=row[6],
                default_value=row[7],
                format_pattern=row[8],
                created_at=datetime.fromisoformat(row[9]),
                updated_at=datetime.fromisoformat(row[10])
            )
            
            # Get category values if this is a categorical variable
            if variable.data_type == DataType.CATEGORICAL:
                cursor.execute("SELECT code, label, description, order_num, is_default FROM category_values WHERE variable_id = ?", (variable_id,))
                category_rows = cursor.fetchall()
                
                variable.categories = []
                for cat_row in category_rows:
                    category = CategoryValue(
                        code=cat_row[0],
                        label=cat_row[1],
                        description=cat_row[2],
                        order=cat_row[3],
                        is_default=bool(cat_row[4])
                    )
                    variable.categories.append(category)
            
            logger.info(f"Retrieved variable {variable.name}")
            return variable
            
        except Exception as e:
            logger.error(f"Error retrieving variable: {str(e)}")
            return None