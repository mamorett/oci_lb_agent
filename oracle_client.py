from typing import List, Dict, Any
from datetime import datetime, timedelta
import json
import oci
from oci import config
from oci.logging import LoggingManagementClient
from oci.loggingsearch import LogSearchClient
from oci.loggingsearch.models import SearchLogsDetails

# Simple import
from models import LogEntry

class OracleLogsClient:
    def __init__(self):
        """Initialize Oracle Cloud connection only"""
        try:
            # Initialize Oracle Cloud clients
            self.config = oci.config.from_file(profile_name="SWEDEN")
            self.logging_client = LoggingManagementClient(self.config)
            self.search_client = LogSearchClient(self.config)
            
            # Oracle Cloud identifiers from your log structure
            self.compartment_id = "ocid1.tenancy.oc1..aaaaaaaafpspkvejmkhbeuwq7cdfwvfl2skhgjzrpsegy6ppgf5ijx5skmqq"
            self.log_group_id = "ocid1.loggroup.oc1.eu-stockholm-1.amaaaaaarzyg5cyalxcetb2kpp77z5dungbvwwx6zxrv6oqsfkptb4yrtbna"
            self.log_id = "ocid1.log.oc1.eu-stockholm-1.amaaaaaarzyg5cyamgcm3ulhegg5x6qf7mslsr3z2hhitsm7rf3hdf5vf6iq"
            
            print("✅ Oracle Cloud connection initialized successfully")
            print(f"📋 Targeting log: {self.log_id}")
            
        except Exception as e:
            print(f"❌ Failed to initialize Oracle Cloud connection: {e}")
            raise
    
    def _build_base_query(self) -> str:
        """Build the base query targeting the specific log"""
        return f'search "{self.compartment_id}/{self.log_group_id}/{self.log_id}"'
    
    def _build_country_query(self, params: Dict[str, Any]) -> str:
        base_query = self._build_base_query()
        conditions = []
        
        if params.get('country'):
            conditions.append(f"data.Country = '{params['country']}'")
        
        if params.get('country_code'):
            conditions.append(f"data.CountryCode = '{params['country_code']}'")
        
        if conditions:
            base_query += ' | where ' + ' and '.join(conditions)
        
        if params.get('limit'):
            base_query += f" | limit {params['limit']}"
            
        return base_query

    def _build_location_query(self, params: Dict[str, Any]) -> str:
        query = self._build_base_query()
        query += f" | where data.Latitude >= {params['lat_min']} and data.Latitude <= {params['lat_max']}"
        query += f" | where data.Longitude >= {params['lon_min']} and data.Longitude <= {params['lon_max']}"
        
        if params.get('limit'):
            query += f" | limit {params['limit']}"
            
        return query

    def _build_ip_query(self, params: Dict[str, Any]) -> str:
        query = self._build_base_query()
        
        if params.get('ip_address'):
            query += f" | where data.IP = '{params['ip_address']}'"
        elif params.get('ip_range'):
            ip_prefix = params["ip_range"].split('/')[0].rsplit('.', 1)[0]
            query += f" | where data.IP like '{ip_prefix}%'"
        
        if params.get('limit'):
            query += f" | limit {params['limit']}"
            
        return query

    def _build_protocol_query(self, protocol: str, params: Dict[str, Any]) -> str:
        query = self._build_base_query()
        query += f" | where data.Protocol = '{protocol}'"
        
        if params.get('limit'):
            query += f" | limit {params['limit']}"
            
        return query

    def _build_isp_query(self, isp: str, params: Dict[str, Any]) -> str:
        query = self._build_base_query()
        query += f" | where data.ISP = '{isp}'"
        
        if params.get('limit'):
            query += f" | limit {params['limit']}"
            
        return query

    async def _execute_oracle_query(
        self, 
        query: str, 
        start_time: datetime, 
        end_time: datetime, 
        max_results: int = None
    ) -> List[Dict]:
        """Execute the actual Oracle Cloud Logging query with pagination support"""
        try:
            print(f"🔍 Executing query: {query}")
            print(f"📅 Time range: {start_time} to {end_time}")

            search_details = SearchLogsDetails(
                time_start=start_time,
                time_end=end_time,
                search_query=query,
                is_return_field_info=False
            )

            all_logs = []
            next_page = None

            while True:
                response = self.search_client.search_logs(
                    search_logs_details=search_details,
                    page=next_page
                )

                for result in response.data.results:
                    try:
                        log_data = json.loads(result.data) if isinstance(result.data, str) else result.data
                        all_logs.append(log_data)
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse log JSON: {e}")
                        continue

                # Pagination: get next page token
                next_page = response.headers.get('opc-next-page')
                if not next_page:
                    break

                # Stop if we've reached max_results
                if max_results and len(all_logs) >= max_results:
                    all_logs = all_logs[:max_results]
                    break

            print(f"✅ Found {len(all_logs)} log entries (with pagination)")
            return all_logs

        except Exception as e:
            print(f"❌ Error executing Oracle query: {e}")
            return []

    def _parse_oracle_log_entry(self, oracle_log: Dict) -> LogEntry:
        """Parse Oracle log JSON into LogEntry model"""
        try:
            log_content = oracle_log.get('logContent', {})
            data = log_content.get('data', {})
            
            # Convert timestamp - Oracle gives milliseconds since epoch
            timestamp_ms = oracle_log.get('datetime', 0)
            timestamp = datetime.fromtimestamp(timestamp_ms / 1000.0)
            
            return LogEntry(
                timestamp=timestamp,
                ip=data.get('IP', ''),
                protocol=data.get('Protocol', ''),
                latitude=float(data.get('Latitude', 0.0)),
                longitude=float(data.get('Longitude', 0.0)),
                country=data.get('Country', ''),
                country_code=data.get('CountryCode', ''),
                city=data.get('City', ''),
                isp=data.get('ISP', '')
            )
            
        except Exception as e:
            print(f"Error parsing log entry: {e}")
            return None

    async def search_logs_by_country(self, params: Dict[str, Any]) -> List[LogEntry]:
        """Search logs by country or country code"""
        start_time, end_time = self._parse_time_range(params.get('time_range', '24h'))
        query = self._build_country_query(params)
        max_results = params.get('max_results')

        oracle_logs = await self._execute_oracle_query(query, start_time, end_time, max_results=max_results)
        
        log_entries = []
        for oracle_log in oracle_logs:
            entry = self._parse_oracle_log_entry(oracle_log)
            if entry:
                log_entries.append(entry)
        
        return log_entries

    async def search_logs_by_location(self, params: Dict[str, Any]) -> List[LogEntry]:
        """Search logs within geographic bounds"""
        start_time, end_time = self._parse_time_range(params.get('time_range', '24h'))
        query = self._build_location_query(params)
        max_results = params.get('max_results')

        oracle_logs = await self._execute_oracle_query(query, start_time, end_time, max_results=max_results)
        
        log_entries = []
        for oracle_log in oracle_logs:
            entry = self._parse_oracle_log_entry(oracle_log)
            if entry:
                log_entries.append(entry)
        
        return log_entries

    async def search_logs_by_ip(self, params: Dict[str, Any]) -> List[LogEntry]:
        """Search logs by IP address or range"""
        start_time, end_time = self._parse_time_range(params.get('time_range', '24h'))
        query = self._build_ip_query(params)
        max_results = params.get('max_results')

        oracle_logs = await self._execute_oracle_query(query, start_time, end_time, max_results=max_results)
        
        log_entries = []
        for oracle_log in oracle_logs:
            entry = self._parse_oracle_log_entry(oracle_log)
            if entry:
                log_entries.append(entry)
        
        return log_entries

    async def get_traffic_analytics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get aggregated traffic statistics"""
        start_time, end_time = self._parse_time_range(params.get('time_range', '24h'))
        base_query = self._build_base_query()
        if params.get('limit'):
            base_query += f' | limit {params["limit"] * 10}'  # Get more for analytics
        max_results = params.get('max_results')

        oracle_logs = await self._execute_oracle_query(base_query, start_time, end_time, max_results=max_results)
        
        analytics = self._process_analytics(oracle_logs, params.get('group_by', 'country'))
        analytics['time_range'] = params.get('time_range', '24h')
        analytics['total_requests'] = len(oracle_logs)
        analytics['log_source'] = self.log_id
        
        return analytics

    def _process_analytics(self, oracle_logs: List[Dict], group_by: str) -> Dict[str, Any]:
        """Process logs into analytics summary"""
        from collections import Counter
        
        unique_ips = set()
        grouped_data = []
        protocols = []
        countries = []
        cities = []
        isps = []
        
        for oracle_log in oracle_logs:
            try:
                data = oracle_log.get('logContent', {}).get('data', {})
                
                unique_ips.add(data.get('IP', ''))
                protocols.append(data.get('Protocol', ''))
                countries.append(data.get('Country', ''))
                cities.append(data.get('City', ''))
                isps.append(data.get('ISP', ''))
                
                # Group by requested field
                if group_by == 'country':
                    grouped_data.append(data.get('Country', 'Unknown'))
                elif group_by == 'city':
                    grouped_data.append(f"{data.get('City', 'Unknown')}, {data.get('Country', '')}")
                elif group_by == 'isp':
                    grouped_data.append(data.get('ISP', 'Unknown'))
                elif group_by == 'protocol':
                    grouped_data.append(data.get('Protocol', 'Unknown'))
                    
            except Exception as e:
                print(f"Error processing log for analytics: {e}")
                continue
        
        grouped_counter = Counter(grouped_data)
        protocol_counter = Counter(protocols)
        
        return {
            'unique_ips': len(unique_ips),
            'unique_countries': len(set(countries)),
            'unique_cities': len(set(cities)),
            f'top_{group_by}': [
                {'name': item, 'count': count} 
                for item, count in grouped_counter.most_common(10)
            ],
            'protocol_distribution': dict(protocol_counter.most_common()),
            'top_isps': [isp for isp, _ in Counter(isps).most_common(5)]
        }
    
    def _parse_time_range(self, time_range: str) -> tuple[datetime, datetime]:
        """Parse time range string like '24h', '7d', '1w' into datetime objects"""
        now = datetime.utcnow()
        
        if time_range.endswith('h'):
            hours = int(time_range[:-1])
            start_time = now - timedelta(hours=hours)
        elif time_range.endswith('d'):
            days = int(time_range[:-1])
            start_time = now - timedelta(days=days)
        elif time_range.endswith('w'):
            weeks = int(time_range[:-1])
            start_time = now - timedelta(weeks=weeks)
        else:
            start_time = now - timedelta(hours=24)
            
        return start_time, now
