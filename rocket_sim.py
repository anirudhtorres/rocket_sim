import numpy as np
import matplotlib.pyplot as plt
import time

"""
 The sim is a 3dof point mass rocket simulation. The state is expressed in the NED frame.
"""
# Earth constants
g_0 = 9.80665  # m/s^2
earth_mean_radius = 6378000  # m

# atmospheric parameters
R_specific = 287.053  # J/kg-K
specific_heat_ratio = 1.4

default_atmospheric_params = {'height': [0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0],  # m
                              # Pa
                              'pressure': [101325.0, 22632.10, 5474.89, 868.02, 110.91, 66.94, 3.96],
                              # K
                              'temperature': [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65],
                              'lapse_rate': [-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002]}  # K/m


def atmospheric_model(height, use_standard=True, base_height=default_atmospheric_params['height'][0],
                      base_temp=default_atmospheric_params['temperature'][0], base_pressure=default_atmospheric_params['pressure'][0]):
    if height < base_height:
        return {'success': False}

    if height >= 85.0e3:
        return _thermosphere(height)

    atmospheric_params = default_atmospheric_params

    # find the section of the atmospheric table that applies
    base_heights = atmospheric_params['height']
    index = np.nonzero(np.array(base_heights) <= height)[0][-1]

    base_height = base_heights[index]
    base_pressure = atmospheric_params['pressure'][index]
    base_temp = atmospheric_params['temperature'][index]
    lapse_rate = atmospheric_params['lapse_rate'][index]

    temperature, pressure = get_temperature_pressure_at_height(
        height, lapse_rate, base_height, base_temp, base_pressure)

    density = pressure / (R_specific * temperature)
    speed_of_sound = np.sqrt(
        specific_heat_ratio * R_specific * temperature)

    dynamic_viscosity = calculate_dynamic_viscosity_air(temperature)

    output = {'success': True, 'temperature': temperature,
              'pressure': pressure, 'density': density,
              'speed_of_sound': speed_of_sound, 'dynamic_viscosity': dynamic_viscosity}
    return output


def calculate_dynamic_viscosity_air(temperature):

    reference_viscosity = 1.458e-6
    reference_temperature = 110.4

    return (reference_viscosity * temperature ** 1.5) / (temperature + reference_temperature)


def get_temperature_pressure_at_height(height, lapse_rate, base_height, base_temp, base_pressure):
    if lapse_rate == 0.0:
        temperature = base_temp
        pressure = base_pressure * \
            np.exp(-g_0 * (height - base_height) /
                   (R_specific * base_temp))
    else:
        temperature = base_temp + lapse_rate * (height - base_height)
        pressure = base_pressure * \
            (base_temp / temperature)**(g_0 /
                                        (R_specific * lapse_rate))

    return temperature, pressure


def _thermosphere(height):
    base_heights = [85000, 91000, 110000, 120000]

    if height < base_heights[0]:
        print("Should not call this function for altitudes lower than 85 km! Using `atmosphere(height)'.")
        return atmospheric_model(height)

    if height >= base_heights[3]:
        output = {'success': True, 'temperature': 0.0, 'pressure': 0.0,
                  'density': 0.0, 'speed_of_sound': 1.0e-6, 'dynamic_viscosity': 0.0}
        return output

    if height < base_heights[1]:
        temperature = 186.8673
    elif height < base_heights[2]:
        temperature = 263.1905 - 76.3232 * \
            np.sqrt(1.0 - ((height - base_heights[1])/-19.9429e3)**2)
    else:
        temperature = 240.0 + 12.0e-3 * (height - base_heights[2])

    # pressure follows the same pattern is entire thermosphere
    height_in_km = height / 1.0e3
    pressure = np.exp(-0.0000000422012 * (height_in_km)**5 + 0.0000213489 * (height_in_km)**4 -
                      0.00426388 * (height_in_km)**3 + 0.421404 * (height_in_km)**2 - 20.8270 * (height_in_km) + 416.225)

    density = pressure / (R_specific * temperature)
    speed_of_sound = np.sqrt(
        specific_heat_ratio * R_specific * temperature)
    dynamic_viscosity = calculate_dynamic_viscosity_air(temperature)

    output = {'success': True, 'temperature': temperature,
              'pressure': pressure, 'density': density,
              'speed_of_sound': speed_of_sound, 'dynamic_viscosity': dynamic_viscosity}

    return output


def rot_x(theta):
    return np.array([[1., 0., 0.], [0., np.cos(theta), np.sin(theta)], [0., -np.sin(theta), np.cos(theta)]])


def rot_y(theta):
    return np.array([[np.cos(theta), 0., -np.sin(theta)], [0., 1., 0.], [np.sin(theta), 0., np.cos(theta)]])


def rot_z(theta):
    return np.array([[np.cos(theta), np.sin(theta), 0.], [-np.sin(theta), np.cos(theta), 0]], [0., 0., 1.])


class RocketSim:
    def __init__(self):
        self.time_final = 200  # seconds
        self.dt = 0.01
        self.no_of_points = int(np.floor(self.time_final/self.dt))
        self.time = np.linspace(0, self.time_final, self.no_of_points)

        # x, y, z, x_dot, y_dot, z_dot, m. This state is expressed in the NED frame.
        self.state = np.zeros((self.no_of_points, 7))
        self.theta = np.zeros((self.no_of_points, 1))
        self.thrust = np.zeros((self.no_of_points, 3))
        self.gravity = np.zeros((self.no_of_points, 3))
        self.drag = np.zeros((self.no_of_points, 3))
        self.alpha = np.zeros((self.no_of_points, 1))

        self.theta_start = 89 * np.pi/180
        self.theta_end = 0 * np.pi/180

        # Engine parameters
        self.Isp = 290  # seconds
        self.m_dot = 1  # kg/s
        self.no_of_engines = 1

        # Rocket aerodynamic properties
        self.cd = 0.5
        self.area = 1  # m^2 (cross sectional area)

        # Rocket mass properties
        self.dry_mass = 10  # kg
        self.fuel = 90  # kg
        self.state[0, 6] = self.dry_mass + self.fuel
        self.time_final_burn = self.fuel/self.m_dot

    def get_thrust_per_engine(self, m_dot):
        # to do with engine parameters
        return self.Isp * g_0 * m_dot

    def get_thrust_force(self, m_dot, C_B2N, switch_on):
        if switch_on:
            thrust = self.get_thrust_per_engine(m_dot)
            return np.matmul(C_B2N, np.array([self.no_of_engines * thrust, 0., 0.]))
        else:
            return np.array([0., 0., 0.])

    def get_drag(self, timestep):
        velocity = np.sqrt(self.state[timestep, 3]
                           ** 2 + self.state[timestep, 5]**2)
        alt = -self.state[timestep, 2]
        atm_model = atmospheric_model(alt)

        drag = self.cd * self.area * atm_model['density'] * velocity**2
        return drag

    def get_drag_force(self, timestep, C_B2N, C_V2B, switch_on):
        if switch_on:
            drag = self.get_drag(timestep)
            return np.matmul(C_B2N, np.matmul(C_V2B, np.array([-drag, 0, 0])))
        else:
            return np.array([0., 0., 0.])

    def get_gravity(self, timestep):
        alt = -self.state[timestep, 2]
        return g_0 * (earth_mean_radius/(earth_mean_radius + alt))**2

    def get_gravity_force(self, timestep, switch_on):
        if switch_on:
            gravity = self.get_gravity(timestep)
            return np.array([0., 0., self.state[timestep, 6]*gravity])
        else:
            return np.array([0., 0., 0.])

    def get_C_N2B(self, theta):
        return rot_y(theta)

    def get_beta(self, timestep):
        return np.arctan2(-self.state[timestep, 5], self.state[timestep, 3])

    def get_theta(self, timestep):
        # Using linear tangent law
        cur_time = self.time[timestep]
        if cur_time <= self.time_final_burn:
            # print(np.tan(self.theta_start) - (np.tan(self.theta_start) -
            #       np.tan(self.theta_end))/self.time_final_burn * cur_time)
            # print(cur_time, self.time_final_burn)

            return np.arctan2(np.tan(self.theta_start) - (np.tan(self.theta_start) - np.tan(self.theta_end))/self.time_final_burn * cur_time, 1)
        else:
            return self.theta_end

    def get_alpha(self, theta, timestep):
        beta = self.get_beta(timestep)
        return theta - beta

    def get_C_V2B(self, theta, timestep):
        alpha = self.get_alpha(theta, timestep)
        self.alpha[timestep] = alpha
        return rot_y(alpha)

    def propogate_state(self, timestep):
        self.theta[timestep] = self.get_theta(timestep)

        if self.state[timestep, 6] <= self.dry_mass:
            # Exhausted the fuel
            m_dot = 0.0
        else:
            m_dot = -self.m_dot

        # Testing
        # print(self.theta[timestep])
        C_B2N = self.get_C_N2B(self.theta[timestep, 0]).T

        C_V2B = self.get_C_V2B(self.theta[timestep], timestep)
        self.drag[timestep] = self.get_drag_force(
            timestep, C_B2N, C_V2B, True)
        # print(self.drag[timestep])
        self.gravity[timestep] = self.get_gravity_force(timestep, True)

        self.thrust[timestep] = self.get_thrust_force(-m_dot, C_B2N, True)

        acc = (self.drag[timestep] + self.gravity[timestep] + self.thrust[timestep]) / \
            self.state[timestep, 6]

        return np.array([self.state[timestep, 3], self.state[timestep, 4], self.state[timestep, 5], acc[0], acc[1], acc[2], m_dot])

    def plot_figs(self):
        # X plotting
        plt.figure()

        plt.subplot(1, 2, 1)
        plt.xlabel("Time (s)")
        plt.ylabel("North (m)")
        plt.plot(self.time, self.state[:, 0])

        # Altitude plotting

        plt.subplot(1, 2, 2)
        plt.xlabel("Time (s)")
        plt.ylabel("Altitude (m)")
        plt.plot(self.time, -self.state[:, 2])

        # X vs altitude
        plt.figure()
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.plot(self.state[:, 0], -self.state[:, 2])

        # Theta plotting
        plt.figure()
        plt.xlabel("Time (s)")
        plt.ylabel("Theta (deg)")
        plt.plot(self.time, self.theta*180/np.pi)

        # Vel north plotting
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.xlabel("Time (s)")
        plt.ylabel("North vel(m/s)")
        plt.plot(self.time, self.state[:, 3])

        # Vel down plotting
        plt.subplot(1, 2, 2)
        plt.xlabel("Time (s)")
        plt.ylabel("North down(m/s)")
        plt.plot(self.time, self.state[:, 5])

        # Mass plotting
        plt.figure()
        plt.xlabel("Time (s)")
        plt.ylabel("Mass (kg)")
        plt.plot(self.time, self.state[:, 6])

        # Thrust
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.plot(self.time, self.thrust[:, 0])
        plt.xlabel("Time (s)")
        plt.ylabel("Thrust x (N)")

        plt.subplot(1, 3, 2)
        plt.plot(self.time, self.thrust[:, 1])
        plt.xlabel("Time (s)")
        plt.ylabel("Thrust y (N)")

        plt.subplot(1, 3, 3)
        plt.plot(self.time, self.thrust[:, 2])
        plt.xlabel("Time (s)")
        plt.ylabel("Thrust z (N)")

        # Gravity
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.plot(self.time, self.gravity[:, 0])
        plt.xlabel("Time (s)")
        plt.ylabel("Gravity x (N)")

        plt.subplot(1, 3, 2)
        plt.plot(self.time, self.gravity[:, 1])
        plt.xlabel("Time (s)")
        plt.ylabel("Gravity y (N)")

        plt.subplot(1, 3, 3)
        plt.plot(self.time, self.gravity[:, 2])
        plt.xlabel("Time (s)")
        plt.ylabel("Gravity z (N)")

        # Drag
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.plot(self.time, self.drag[:, 0])
        plt.xlabel("Time (s)")
        plt.ylabel("Drag x (N)")

        plt.subplot(1, 3, 2)
        plt.plot(self.time, self.drag[:, 1])
        plt.xlabel("Time (s)")
        plt.ylabel("Drag y (N)")

        plt.subplot(1, 3, 3)
        plt.plot(self.time, self.drag[:, 2])
        plt.xlabel("Time (s)")
        plt.ylabel("Drag z (N)")

        # angle of attack
        plt.figure()
        plt.plot(self.time, self.alpha*180/np.pi)
        plt.xlabel('Time (s)')
        plt.ylabel("Angle of attack (deg)")
        plt.show()

    def run_sim(self):
        timestep = 0
        while timestep < self.no_of_points-1:
            x_dot = self.propogate_state(timestep)
            # print("X dot", x_dot)
            self.state[timestep+1] = self.state[timestep] + \
                x_dot*self.dt
            # print("state", self.state[timestep])
            timestep = timestep + 1

        self.plot_figs()


def main():
    rsim = RocketSim()
    rsim.run_sim()


if __name__ == '__main__':
    main()