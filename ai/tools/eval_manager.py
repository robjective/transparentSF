#!/usr/bin/env python3
"""
Evaluation Manager for TransparentSF
Handles database operations for evals, eval groups, and eval results.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, date
from .db_utils import get_postgres_connection, execute_with_connection

class DateAwareJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles date and datetime objects."""
    def default(self, obj):
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        return super().default(obj)

logger = logging.getLogger(__name__)

def get_eval_groups(category: str = None, active_only: bool = True) -> Dict[str, Any]:
    """
    Get eval groups with optional filtering.
    
    Args:
        category: Filter by category
        active_only: Only return active groups
        
    Returns:
        dict: Result with status and list of eval groups
    """
    try:
        connection = get_postgres_connection()
        if not connection:
            return {"status": "error", "message": "Database connection failed"}
        
        cursor = connection.cursor()
        
        query = "SELECT id, name, description, category, is_active, created_at, updated_at FROM eval_groups"
        params = []
        
        where_conditions = []
        if category:
            where_conditions.append("category = %s")
            params.append(category)
        
        if active_only:
            where_conditions.append("is_active = TRUE")
        
        if where_conditions:
            query += " WHERE " + " AND ".join(where_conditions)
        
        query += " ORDER BY name"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        groups = []
        for row in rows:
            groups.append({
                "id": row[0],
                "name": row[1],
                "description": row[2],
                "category": row[3],
                "is_active": row[4],
                "created_at": row[5].isoformat() if row[5] else None,
                "updated_at": row[6].isoformat() if row[6] else None
            })
        
        return {
            "status": "success",
            "count": len(groups),
            "groups": groups
        }
        
    except Exception as e:
        logger.error(f"Error getting eval groups: {str(e)}")
        return {"status": "error", "message": str(e)}
    finally:
        if connection:
            connection.close()

def get_evals(group_id: int = None, active_only: bool = True) -> Dict[str, Any]:
    """
    Get evals with optional filtering.
    
    Args:
        group_id: Filter by group ID
        active_only: Only return active evals
        
    Returns:
        dict: Result with status and list of evals
    """
    try:
        connection = get_postgres_connection()
        if not connection:
            return {"status": "error", "message": "Database connection failed"}
        
        cursor = connection.cursor()
        
        query = """
            SELECT e.id, e.group_id, eg.name as group_name, e.name, e.description, 
                   e.prompt, e.success_criteria, e.success_type, e.expected_tool_calls,
                   e.expected_outputs, e.difficulty_level, e.estimated_time_minutes,
                   e.is_active, e.created_at, e.updated_at
            FROM evals e
            JOIN eval_groups eg ON e.group_id = eg.id
        """
        params = []
        
        where_conditions = []
        if group_id:
            where_conditions.append("e.group_id = %s")
            params.append(group_id)
        
        if active_only:
            where_conditions.append("e.is_active = TRUE")
        
        if where_conditions:
            query += " WHERE " + " AND ".join(where_conditions)
        
        query += " ORDER BY eg.name, e.name"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        evals = []
        for row in rows:
            evals.append({
                "id": row[0],
                "group_id": row[1],
                "group_name": row[2],
                "name": row[3],
                "description": row[4],
                "prompt": row[5],
                "success_criteria": row[6],
                "success_type": row[7],
                "expected_tool_calls": row[8] if row[8] else [],
                "expected_outputs": row[9] if row[9] else [],
                "difficulty_level": row[10],
                "estimated_time_minutes": row[11],
                "is_active": row[12],
                "created_at": row[13].isoformat() if row[13] else None,
                "updated_at": row[14].isoformat() if row[14] else None
            })
        
        return {
            "status": "success",
            "count": len(evals),
            "evals": evals
        }
        
    except Exception as e:
        logger.error(f"Error getting evals: {str(e)}")
        return {"status": "error", "message": str(e)}
    finally:
        if connection:
            connection.close()

def get_eval_by_id(eval_id: int) -> Dict[str, Any]:
    """
    Get a specific eval by ID.
    
    Args:
        eval_id: The eval ID
        
    Returns:
        dict: Result with status and eval details
    """
    try:
        connection = get_postgres_connection()
        if not connection:
            return {"status": "error", "message": "Database connection failed"}
        
        cursor = connection.cursor()
        
        query = """
            SELECT e.id, e.group_id, eg.name as group_name, e.name, e.description, 
                   e.prompt, e.success_criteria, e.success_type, e.expected_tool_calls,
                   e.expected_outputs, e.difficulty_level, e.estimated_time_minutes,
                   e.is_active, e.created_at, e.updated_at
            FROM evals e
            JOIN eval_groups eg ON e.group_id = eg.id
            WHERE e.id = %s
        """
        
        cursor.execute(query, (eval_id,))
        row = cursor.fetchone()
        
        if not row:
            return {"status": "error", "message": "Eval not found"}
        
        eval_data = {
            "id": row[0],
            "group_id": row[1],
            "group_name": row[2],
            "name": row[3],
            "description": row[4],
            "prompt": row[5],
            "success_criteria": row[6],
            "success_type": row[7],
            "expected_tool_calls": row[8] if row[8] else [],
            "expected_outputs": row[9] if row[9] else [],
            "difficulty_level": row[10],
            "estimated_time_minutes": row[11],
            "is_active": row[12],
            "created_at": row[13].isoformat() if row[13] else None,
            "updated_at": row[14].isoformat() if row[14] else None
        }
        
        return {
            "status": "success",
            "eval": eval_data
        }
        
    except Exception as e:
        logger.error(f"Error getting eval by ID: {str(e)}")
        return {"status": "error", "message": str(e)}
    finally:
        if connection:
            connection.close()

def create_eval_group(name: str, description: str, category: str) -> Dict[str, Any]:
    """
    Create a new eval group.
    
    Args:
        name: Group name
        description: Group description
        category: Group category
        
    Returns:
        dict: Result with status and group ID
    """
    try:
        connection = get_postgres_connection()
        if not connection:
            return {"status": "error", "message": "Database connection failed"}
        
        cursor = connection.cursor()
        
        query = """
            INSERT INTO eval_groups (name, description, category)
            VALUES (%s, %s, %s)
            RETURNING id
        """
        
        cursor.execute(query, (name, description, category))
        group_id = cursor.fetchone()[0]
        connection.commit()
        
        return {
            "status": "success",
            "group_id": group_id,
            "message": f"Created eval group '{name}' with ID {group_id}"
        }
        
    except Exception as e:
        logger.error(f"Error creating eval group: {str(e)}")
        return {"status": "error", "message": str(e)}
    finally:
        if connection:
            connection.close()

def update_eval_group(group_id: int, updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update an eval group.
    
    Args:
        group_id: Group ID
        updates: Dictionary of fields to update
        
    Returns:
        dict: Result with status and message
    """
    try:
        connection = get_postgres_connection()
        if not connection:
            return {"status": "error", "message": "Database connection failed"}
        
        cursor = connection.cursor()
        
        # Build update query dynamically
        set_clauses = []
        params = []
        
        allowed_fields = ['name', 'description', 'category', 'is_active']
        for field, value in updates.items():
            if field in allowed_fields:
                set_clauses.append(f"{field} = %s")
                params.append(value)
        
        if not set_clauses:
            return {"status": "error", "message": "No valid fields to update"}
        
        params.append(group_id)
        query = f"""
            UPDATE eval_groups 
            SET {', '.join(set_clauses)}
            WHERE id = %s
        """
        
        cursor.execute(query, params)
        
        if cursor.rowcount == 0:
            return {"status": "error", "message": "Eval group not found"}
        
        connection.commit()
        
        return {
            "status": "success",
            "message": f"Updated eval group with ID {group_id}"
        }
        
    except Exception as e:
        logger.error(f"Error updating eval group: {str(e)}")
        return {"status": "error", "message": str(e)}
    finally:
        if connection:
            connection.close()

def delete_eval_group(group_id: int) -> Dict[str, Any]:
    """
    Delete an eval group (and all its evals).
    
    Args:
        group_id: Group ID
        
    Returns:
        dict: Result with status and message
    """
    try:
        connection = get_postgres_connection()
        if not connection:
            return {"status": "error", "message": "Database connection failed"}
        
        cursor = connection.cursor()
        
        # Check if group exists
        cursor.execute("SELECT id FROM eval_groups WHERE id = %s", (group_id,))
        if not cursor.fetchone():
            return {"status": "error", "message": "Eval group not found"}
        
        # Delete the group (cascade will handle evals and results)
        cursor.execute("DELETE FROM eval_groups WHERE id = %s", (group_id,))
        connection.commit()
        
        return {
            "status": "success",
            "message": f"Deleted eval group with ID {group_id}"
        }
        
    except Exception as e:
        logger.error(f"Error deleting eval group: {str(e)}")
        return {"status": "error", "message": str(e)}
    finally:
        if connection:
            connection.close()

def create_eval(
    group_id: int,
    name: str,
    description: str,
    prompt: str,
    success_criteria: str,
    success_type: str,
    expected_tool_calls: List[Dict] = None,
    expected_outputs: List[Dict] = None,
    difficulty_level: str = "medium",
    estimated_time_minutes: int = 5
) -> Dict[str, Any]:
    """
    Create a new eval.
    
    Args:
        group_id: Group ID
        name: Eval name
        description: Eval description
        prompt: The prompt to test
        success_criteria: Success criteria description
        success_type: Type of success criteria
        expected_tool_calls: Expected tool calls
        expected_outputs: Expected outputs
        difficulty_level: Difficulty level
        estimated_time_minutes: Estimated time in minutes
        
    Returns:
        dict: Result with status and eval ID
    """
    try:
        connection = get_postgres_connection()
        if not connection:
            return {"status": "error", "message": "Database connection failed"}
        
        cursor = connection.cursor()
        
        query = """
            INSERT INTO evals (group_id, name, description, prompt, success_criteria, 
                              success_type, expected_tool_calls, expected_outputs, 
                              difficulty_level, estimated_time_minutes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        
        cursor.execute(query, (
            group_id, name, description, prompt, success_criteria, success_type,
            json.dumps(expected_tool_calls or []),
            json.dumps(expected_outputs or []),
            difficulty_level, estimated_time_minutes
        ))
        eval_id = cursor.fetchone()[0]
        connection.commit()
        
        return {
            "status": "success",
            "eval_id": eval_id,
            "message": f"Created eval '{name}' with ID {eval_id}"
        }
        
    except Exception as e:
        logger.error(f"Error creating eval: {str(e)}")
        return {"status": "error", "message": str(e)}
    finally:
        if connection:
            connection.close()

def update_eval(eval_id: int, updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update an eval.
    
    Args:
        eval_id: Eval ID
        updates: Dictionary of fields to update
        
    Returns:
        dict: Result with status and message
    """
    try:
        connection = get_postgres_connection()
        if not connection:
            return {"status": "error", "message": "Database connection failed"}
        
        cursor = connection.cursor()
        
        # Build update query dynamically
        set_clauses = []
        params = []
        
        allowed_fields = ['name', 'description', 'prompt', 'success_criteria', 'success_type', 
                         'expected_tool_calls', 'expected_outputs', 'difficulty_level', 
                         'estimated_time_minutes', 'is_active']
        
        for field, value in updates.items():
            if field in allowed_fields:
                if field in ['expected_tool_calls', 'expected_outputs']:
                    set_clauses.append(f"{field} = %s")
                    params.append(json.dumps(value))
                else:
                    set_clauses.append(f"{field} = %s")
                    params.append(value)
        
        if not set_clauses:
            return {"status": "error", "message": "No valid fields to update"}
        
        params.append(eval_id)
        query = f"""
            UPDATE evals 
            SET {', '.join(set_clauses)}
            WHERE id = %s
        """
        
        cursor.execute(query, params)
        
        if cursor.rowcount == 0:
            return {"status": "error", "message": "Eval not found"}
        
        connection.commit()
        
        return {
            "status": "success",
            "message": f"Updated eval with ID {eval_id}"
        }
        
    except Exception as e:
        logger.error(f"Error updating eval: {str(e)}")
        return {"status": "error", "message": str(e)}
    finally:
        if connection:
            connection.close()

def delete_eval(eval_id: int) -> Dict[str, Any]:
    """
    Delete an eval (and all its results).
    
    Args:
        eval_id: Eval ID
        
    Returns:
        dict: Result with status and message
    """
    try:
        connection = get_postgres_connection()
        if not connection:
            return {"status": "error", "message": "Database connection failed"}
        
        cursor = connection.cursor()
        
        # Check if eval exists
        cursor.execute("SELECT id FROM evals WHERE id = %s", (eval_id,))
        if not cursor.fetchone():
            return {"status": "error", "message": "Eval not found"}
        
        # Delete the eval (cascade will handle results)
        cursor.execute("DELETE FROM evals WHERE id = %s", (eval_id,))
        connection.commit()
        
        return {
            "status": "success",
            "message": f"Deleted eval with ID {eval_id}"
        }
        
    except Exception as e:
        logger.error(f"Error deleting eval: {str(e)}")
        return {"status": "error", "message": str(e)}
    finally:
        if connection:
            connection.close()

def save_eval_result(
    eval_id: int,
    model_name: str,
    prompt_used: str,
    response_received: str = None,
    tool_calls_made: List[Dict] = None,
    success_score: float = None,
    success_details: Dict = None,
    execution_time_seconds: float = None,
    error_message: str = None,
    status: str = "completed",
    conversation_history: List[Dict] = None
) -> Dict[str, Any]:
    """
    Save an eval result.
    
    Args:
        eval_id: Eval ID
        model_name: Name of the model used
        prompt_used: The prompt that was used
        response_received: Response from the model
        tool_calls_made: Tool calls that were made
        success_score: Success score (0.0 to 1.0)
        success_details: Detailed success information
        execution_time_seconds: Execution time
        error_message: Error message if any
        status: Result status
        
    Returns:
        dict: Result with status and result ID
    """
    try:
        connection = get_postgres_connection()
        if not connection:
            return {"status": "error", "message": "Database connection failed"}
        
        cursor = connection.cursor()
        
        query = """
            INSERT INTO eval_results (eval_id, model_name, prompt_used, response_received,
                                    tool_calls_made, success_score, success_details,
                                    execution_time_seconds, error_message, status, conversation_history)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        
        cursor.execute(query, (
            eval_id, model_name, prompt_used, response_received,
            json.dumps(tool_calls_made or [], cls=DateAwareJSONEncoder),
            success_score,
            json.dumps(success_details or {}, cls=DateAwareJSONEncoder),
            execution_time_seconds, error_message, status,
            json.dumps(conversation_history or [], cls=DateAwareJSONEncoder)
        ))
        result_id = cursor.fetchone()[0]
        connection.commit()
        
        return {
            "status": "success",
            "result_id": result_id,
            "message": f"Saved eval result with ID {result_id}"
        }
        
    except Exception as e:
        logger.error(f"Error saving eval result: {str(e)}")
        return {"status": "error", "message": str(e)}
    finally:
        if connection:
            connection.close()

def get_eval_results(eval_id: int = None, model_name: str = None, limit: int = 100) -> Dict[str, Any]:
    """
    Get eval results with optional filtering.
    
    Args:
        eval_id: Filter by eval ID
        model_name: Filter by model name
        limit: Maximum number of results
        
    Returns:
        dict: Result with status and list of results
    """
    try:
        connection = get_postgres_connection()
        if not connection:
            return {"status": "error", "message": "Database connection failed"}
        
        cursor = connection.cursor()
        
        query = """
            SELECT er.id, er.eval_id, e.name as eval_name, eg.name as group_name,
                   er.model_name, er.prompt_used, er.response_received, er.tool_calls_made,
                   er.success_score, er.success_details, er.execution_time_seconds,
                   er.error_message, er.status, er.created_at, er.conversation_history
            FROM eval_results er
            JOIN evals e ON er.eval_id = e.id
            JOIN eval_groups eg ON e.group_id = eg.id
        """
        params = []
        
        where_conditions = []
        if eval_id:
            where_conditions.append("er.eval_id = %s")
            params.append(eval_id)
        
        if model_name:
            where_conditions.append("er.model_name = %s")
            params.append(model_name)
        
        if where_conditions:
            query += " WHERE " + " AND ".join(where_conditions)
        
        query += " ORDER BY er.created_at DESC LIMIT %s"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            results.append({
                "id": row[0],
                "eval_id": row[1],
                "eval_name": row[2],
                "group_name": row[3],
                "model_name": row[4],
                "prompt_used": row[5],
                "response_received": row[6],
                "tool_calls_made": row[7] if row[7] else [],
                "success_score": row[8],
                "success_details": row[9] if row[9] else {},
                "execution_time_seconds": row[10],
                "error_message": row[11],
                "status": row[12],
                "created_at": row[13].isoformat() if row[13] else None,
                "conversation_history": row[14] if row[14] else []
            })
        
        return {
            "status": "success",
            "count": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error getting eval results: {str(e)}")
        return {"status": "error", "message": str(e)}
    finally:
        if connection:
            connection.close()

def get_eval_summary() -> Dict[str, Any]:
    """
    Get a summary of eval results.
    
    Returns:
        dict: Result with status and summary statistics
    """
    try:
        connection = get_postgres_connection()
        if not connection:
            return {"status": "error", "message": "Database connection failed"}
        
        cursor = connection.cursor()
        
        # Get overall statistics
        cursor.execute("""
            SELECT 
                COUNT(*) as total_results,
                COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_results,
                COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_results,
                AVG(success_score) as avg_success_score,
                AVG(execution_time_seconds) as avg_execution_time
            FROM eval_results
        """)
        overall_stats = cursor.fetchone()
        
        # Get results by model
        cursor.execute("""
            SELECT 
                model_name,
                COUNT(*) as total_runs,
                AVG(success_score) as avg_score,
                AVG(execution_time_seconds) as avg_time
            FROM eval_results
            WHERE status = 'completed'
            GROUP BY model_name
            ORDER BY avg_score DESC
        """)
        model_stats = cursor.fetchall()
        
        # Get results by eval group
        cursor.execute("""
            SELECT 
                eg.name as group_name,
                COUNT(er.id) as total_runs,
                AVG(er.success_score) as avg_score,
                AVG(er.execution_time_seconds) as avg_time
            FROM eval_groups eg
            LEFT JOIN evals e ON eg.id = e.group_id
            LEFT JOIN eval_results er ON e.id = er.eval_id AND er.status = 'completed'
            GROUP BY eg.id, eg.name
            ORDER BY avg_score DESC NULLS LAST
        """)
        group_stats = cursor.fetchall()
        
        summary = {
            "overall": {
                "total_results": overall_stats[0] or 0,
                "completed_results": overall_stats[1] or 0,
                "failed_results": overall_stats[2] or 0,
                "avg_success_score": float(overall_stats[3]) if overall_stats[3] else 0.0,
                "avg_execution_time": float(overall_stats[4]) if overall_stats[4] else 0.0
            },
            "by_model": [
                {
                    "model_name": row[0],
                    "total_runs": row[1],
                    "avg_score": float(row[2]) if row[2] else 0.0,
                    "avg_time": float(row[3]) if row[3] else 0.0
                }
                for row in model_stats
            ],
            "by_group": [
                {
                    "group_name": row[0],
                    "total_runs": row[1] or 0,
                    "avg_score": float(row[2]) if row[2] else 0.0,
                    "avg_time": float(row[3]) if row[3] else 0.0
                }
                for row in group_stats
            ]
        }
        
        return {
            "status": "success",
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"Error getting eval summary: {str(e)}")
        return {"status": "error", "message": str(e)}
    finally:
        if connection:
            connection.close() 