from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime, UTC
import time
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

class InfluxBase:
    """Provide necessary attributes for database connection"""

    def __init__(self):
        self.write_api = None
        self.query_api = None
        self.url = os.getenv("URL")
        self.token = os.getenv("TOKEN")
        self.org = os.getenv("ORG")
        self.bucket = os.getenv("BUCKET")
        self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org)

    # "__enter__" and "__exit__"  are magical/special methods. These allow this class to become context manager

    def __enter__(self): # This method executes when program uses "with". It usually prepares the DB connection, but this is already done in "__init__"
                         # This method returns an instance to the "variable" that called the class
        return self

    def __exit__(self, exc_type, exc_val, exc_tb): # This method starts at the end of "with" block" regardless of error.
                                                   # It tidies up the work (closes files, database connection etc.)
                                                   # Python automatically sends information to the attributes had any error occurs.
        self.client.close()
        print("Connection closed.")

# The block below is used to "listen" if any key is pressed.
# In this case, if "q" is pressed, Python will declare "stop.set()"
# This will make the while loop in "send_data" method to stop.
# Which means it will stop the loop sending the information and connection will close.
# This is important, because we do not want to keep opening and closing the connection each loop.


class Data_Write(InfluxBase):
    """Send data to database"""

    def __init__(self):
        """Take necessary attributes for database connection from parent class"""
        super().__init__()

    def initial_data(self):
        """Manually creates data""" 

        point = (
            Point("rbmk_reactor_metrics")
            .field("fuel_reactivity", 5.905)
            .field("orm_value", 102.0)
            .field("partially_inserted", 0.0)
            .field("inlet_temp_c", 270.0)
            .field("outlet_temp_c", 284.0)
            .field("coolant_flow_m3h", 45000.0)
            .field("tau", 100.0)
            .field("thermal_power_mw", 3200.0)
            .field("reactivity_delta", 0.0)
            .field("xenon_level", 1.0)
            .field("neutron_flux_pct", 100.0)
            .field("severity_level", 0.8)
            .field("subsystem", "a")
            .field("alarm_message", "b")
            .time(datetime.now(UTC), WritePrecision.NS)
        )

        print(point.to_line_protocol())
        with self.client.write_api(write_options=SYNCHRONOUS) as self.write_api:
            self.send_data(point)


    def generated_data(self, **data):
        """Takes data and converts to InfluxDB scheme"""

        point = Point("rbmk_reactor_metrics")
        for key, value in data.items():
            point = point.field(key, value)

        point = point.time(datetime.now(UTC), WritePrecision.NS)

        self.send_data(point)

    def send_data(self, point):
        """Sends the data to database"""

        try:
            self.write_api.write(bucket=self.bucket, org=self.org, record=point)
            print("Data successfully sent!")
        except Exception as e:
            print("Write error: ", e)

        time.sleep(1)



class Data_Read(InfluxBase):
    """Take data from database"""

    def __init__(self):
        """Take necessary attributes for database connection from parent class"""

        super().__init__()

    def take_data(self, last, time_range):
        """ Take data from the database"""

        self.tables = []
        self.query = f'''
            from(bucket: "{self.bucket}")
              |> range(start: -{time_range})
              |> filter(fn: (r) => r._measurement == "rbmk_reactor_metrics")
              |> filter(fn: (r) =>
                  contains(
                    value: r._field,
                    set: [
                      "fuel_reactivity",
                      "orm_value",
                      "partially_inserted",
                      "inlet_temp_c",
                      "outlet_temp_c",
                      "coolant_flow_m3h",
                      "v_steam",
                      "tau",
                      "thermal_power_mw",
                      "reactivity_delta",
                      "xenon_level",
                      "neutron_flux_pct",
                    ]

                  )

                )
              {last}
        '''
        self.query_api = self.client.query_api()
        try:
            self.tables = self.query_api.query(self.query, org=self.org)
            print("Data successfully taken!")
        except Exception as e:
            print("Write error: ", e)



    def influx_to_df(self):
        """Transform InfluxDB output scheme to dictionary for DataFrame"""

        self.conv_data = {}
        time_list = []

        for table in self.tables: # We do not need to create sophisticated code or utilize libraries, because all tables have one consistent timestamp and there is no tag.
            value_list = []
            if not time_list: # "time_list" collects timestamp for each record. If "time_list" is empty, this will fill the list with timestamps
                              # Since this database has one consistent timestamp for each record, we can collect it only once for the entire loop.

                for record in table.records:
                    dt = record.get_time()
                    time_list.append(dt.strftime("%Y-%m-%d %H:%M:%S"))

            # Code below appends values from table to "value_list" list and field to a single variable "field_key"
            # Then we do dictionary nesting, treating appending value as a list to a single key
            # The dictionary is as follows: field_key (key): value_list (value)

            for record in table.records:
                value_list.append(record.get_value())
                field_key = record.get_field()

            if len(value_list) == 1:
                self.conv_data[f'{field_key}'] = value_list[0]
            else:
                self.conv_data[f'{field_key}'] = value_list

        self.conv_data['time'] = time_list

        self.df = pd.DataFrame(self.conv_data)









#         with Data_Write() as write:
#             write.initial_data()
#             time.sleep(2)
#
#
#

#     with Data_Read() as take:
#         take.take_data(last="", time_range="4d")
#         take.influx_to_df()
#         time.sleep(2)
#         take.df.to_parquet('Influx_RBML_data.parquet')


