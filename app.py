import streamlit as st
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model_filename = 'kddcup.h5'
model = tf.keras.models.load_model(model_filename)

# Mapping for human-readable outcome labels
outcome_mapping = {
    0: "Normal",
    1: "Buffer Overflow",
    2: "Load Module",
    3: "Perl",
    4: "Neptune",
    5: "Smurf",
    6: "Guess Passwd",
    7: "Pod",
    8: "Teardrop",
    9: "Port Sweep",
    10: "IP Sweep",
    11: "Land",
    12: "Ftp Write",
    13: "Back",
    14: "Imap",
    15: "Satan",
    16: "PHF",
    17: "Nmap",
    18: "Multihop",
    19: "Warez Master",
    20: "Warez Client",
    21: "Spy",
    22: "Root Kit"
}

def preprocess_input(data):
    categorical_mapping = {
        "protocol_type": {"tcp": 0, "udp": 1, "icmp": 2},
        "flag": {'SF': 0, 'S1': 1, 'REJ': 2, 'S2': 3, 'S0': 4, 'S3': 5, 'RSTO': 6, 'RSTR': 7, 'RSTOS0': 8, 'OTH': 9, 'SH': 10},
        "service": {'http': 0, 'smtp': 1, 'finger': 2, 'domain_u': 3, 'auth': 4, 'telnet': 5,
    'ftp': 6, 'eco_i': 7, 'ntp_u': 8, 'ecr_i': 9, 'other': 10, 'private': 11,
    'pop_3': 12, 'ftp_data': 13, 'rje': 14, 'time': 15, 'mtp': 16, 'link': 17,
    'remote_job': 18, 'gopher': 19, 'ssh': 20, 'name': 21, 'whois': 22,
    'domain': 23, 'login': 24, 'imap4': 25, 'daytime': 26, 'ctf': 27,
    'nntp': 28, 'shell': 29, 'IRC': 30, 'nnsp': 31, 'http_443': 32,
    'exec': 33, 'printer': 34, 'efs': 35, 'courier': 36, 'uucp': 37,
    'klogin': 38, 'kshell': 39, 'echo': 40, 'discard': 41, 'systat': 42,
    'supdup': 43, 'iso_tsap': 44, 'hostnames': 45, 'csnet_ns': 46, 'pop_2': 47,
    'sunrpc': 48, 'uucp_path': 49, 'netbios_ns': 50, 'netbios_ssn': 51,
    'netbios_dgm': 52, 'sql_net': 53, 'vmnet': 54, 'bgp': 55, 'Z39_50': 56,
    'ldap': 57, 'netstat': 58, 'urh_i': 59, 'X11': 60, 'urp_i': 61,
    'pm_dump': 62, 'tftp_u': 63, 'tim_i': 64, 'red_i': 65}
    }

    for feature, mapping in categorical_mapping.items():
        data[feature] = mapping[data[feature][0]]  # Use [0] to access the selected value

    scaler = MinMaxScaler()
    columns_to_scale = ["src_bytes", "count", "serror_rate"]
    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

    return data


# Main Function
def main():
    st.markdown("<h1 style='text-align: center;'>KDD Cup 1999 Data</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Computer network intrusion detection</p>", unsafe_allow_html=True)
    
    st.subheader('About Dataset')
    st.markdown("<p style='text-align: justify;'>This is the data set used for The Third International Knowledge Discovery and Data Mining Tools Competition, which was held in conjunction with KDD-99 The Fifth International Conference on Knowledge Discovery and Data Mining. The competition task was to build a network intrusion detector, a predictive model capable of distinguishing between bad'' connections, called intrusions or attacks, andgood'' normal connections. This database contains a standard set of data to be audited, which includes a wide variety of intrusions simulated in a military network environment.</p>", unsafe_allow_html=True)

    # Input form
    st.subheader("Input Features")
    protocol_type = st.selectbox("Protocol Type", ["tcp", "udp", "icmp"])
    flag = st.selectbox("Flag", ['SF', 'S1', 'REJ', 'S2', 'S0', 'S3', 'RSTO', 'RSTR', 'RSTOS0', 'OTH', 'SH'])
    service = st.selectbox("Service", ['http', 'smtp', 'finger', 'domain_u', 'auth', 'telnet', 'ftp',
                                       'eco_i', 'ntp_u', 'ecr_i', 'other', 'private', 'pop_3', 'ftp_data',
                                       'rje', 'time', 'mtp', 'link', 'remote_job', 'gopher', 'ssh',
                                       'name', 'whois', 'domain', 'login', 'imap4', 'daytime', 'ctf',
                                       'nntp', 'shell', 'IRC', 'nnsp', 'http_443', 'exec', 'printer',
                                       'efs', 'courier', 'uucp', 'klogin', 'kshell', 'echo', 'discard',
                                       'systat', 'supdup', 'iso_tsap', 'hostnames', 'csnet_ns', 'pop_2',
                                       'sunrpc', 'uucp_path', 'netbios_ns', 'netbios_ssn', 'netbios_dgm',
                                       'sql_net', 'vmnet', 'bgp', 'Z39_50', 'ldap', 'netstat', 'urh_i',
                                       'X11', 'urp_i', 'pm_dump', 'tftp_u', 'tim_i', 'red_i'])

    src_bytes = st.number_input("Source Bytes", min_value=0)
    count = st.number_input("Count", min_value=0)
    serror_rate = st.number_input("Serror Rate", min_value=0.0, max_value=1.0, step=0.01)

    input_data = pd.DataFrame({
    "protocol_type": [protocol_type],
    "flag": [flag],
    "service": [service],
    "src_bytes": [src_bytes],
    "count": [count],
    "serror_rate": [serror_rate]
})


    # Preprocess input
    input_data = preprocess_input(input_data)

    if st.button("Predict"):
        prediction = model.predict(input_data)
        predicted_class_idx = prediction.argmax()

        if predicted_class_idx in outcome_mapping:
            outcome_label = outcome_mapping[predicted_class_idx]
            st.subheader("Prediction Result")
            st.success(f"🎉 The predicted outcome is {outcome_label}.")
        else:
            st.error("Unable to determine predicted outcome.")

if __name__ == "__main__":
    main()
