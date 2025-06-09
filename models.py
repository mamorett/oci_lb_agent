from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional, Dict, Any

class LogEntry(BaseModel):
    timestamp: datetime
    ip: str
    protocol: str
    latitude: float
    longitude: float
    country: str
    country_code: str
    city: str
    isp: str

class LogSearchParams(BaseModel):
    time_range: Optional[str] = "24h"
    limit: Optional[int] = 100

class CountrySearchParams(LogSearchParams):
    country: Optional[str] = None
    country_code: Optional[str] = None

class LocationSearchParams(LogSearchParams):
    lat_min: Optional[float] = None
    lat_max: Optional[float] = None
    lon_min: Optional[float] = None
    lon_max: Optional[float] = None

class IPSearchParams(LogSearchParams):
    ip_address: Optional[str] = None
    ip_range: Optional[str] = None

class AnalyticsParams(LogSearchParams):
    group_by: str = "country"  # country, city, isp, protocol
    
class LogResponse(BaseModel):
    logs: List[LogEntry]
    total_count: int
    summary: Dict[str, Any]
