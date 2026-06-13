from schemas import Reactor, reactor_run
import database as db
from influxdb_client.client.write_api import SYNCHRONOUS



def main():
    last = "|> last()"
    time_range = "5d"
    read = db.Data_Read()
    read.take_data(last, time_range)
    read.influx_to_df()
    reactor = Reactor(**read.conv_data)
    with db.Data_Write() as write:
        with write.client.write_api(write_options=SYNCHRONOUS) as write.write_api:
            while True:
                reactor_run(reactor, 1.0)
                print(f"Thermal power main: {reactor.thermal_power_mw}")
                print("\n")
                data = {
                    "thermal_power_mw": reactor.thermal_power_mw,
                    "fuel_reactivity": reactor.fuel_reactivity,
                    "orm_value": reactor.orm_value,
                    "partially_inserted": reactor.partially_inserted,
                    "inlet_temp_c": reactor.inlet_temp_c,
                    "outlet_temp_c": reactor.outlet_temp_c,
                    "coolant_flow_m3h": reactor.coolant_flow_m3h,
                    "v_steam": reactor.v_steam,
                    "tau": reactor.tau,
                    "reactivity_delta": reactor.reactivity_delta,
                    "xenon_level": reactor.xenon_level,
                    "neutron_flux_pct": reactor.neutron_flux_pct,
                    # "severity_level": reactor.severity_level,
                    # "subsystem": reactor.subsystem,
                    # "alarm_message": reactor.alarm_message,
                }
                write.generated_data(**data)



    #pętla programu +
    #połączenie z bazą danych +
    #pobranie danych z bazy danych +
    #inicjalizacja reaktora co 10sek nowy log
    #przy przekroczeniu progu reaktywności automatycznie dodanie control rods +
    #powyżej jakiejś temperatury zwiększenie przepływu chłodziwa
    #Alarmujące wartości
    #user input po wciśnięciu jakiegoś przycisku w terminalu +
    #po wpisaniu AZ5 wsunięcie wszystkich prętów kontrolnych
if __name__ == "__main__":
    main()