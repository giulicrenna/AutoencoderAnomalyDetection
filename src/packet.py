import pandas as pd
import ipaddress
import numpy as np
from typing import List, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class PacketParser:
    """
    A utility class for preprocessing and vectorizing packet data from network traffic.
    Handles IP normalization, protocol encoding, feature matrix construction, and PCA transformation.
    """

    def __init__(self, df: pd.DataFrame, forced_protocols: Optional[List[str]] = None) -> None:
        """
        Initialize the parser with a DataFrame and optional fixed protocol list.

        Parameters:
            df (pd.DataFrame): Raw packet data.
            forced_protocols (List[str], optional): Predefined list of protocols to encode.
        """
        self.df = df.copy()
        self._clean_data()
        self.protocols = forced_protocols if forced_protocols else self._extract_unique_protocols()
        self._scaler: Optional[StandardScaler] = None
        self._pca: Optional[PCA] = None

    def _clean_data(self) -> None:
        """
        Clean and normalize the dataset:
        - Drop rows with missing essential fields
        - Normalize time between 0 and 1
        - Detect whether source/destination IPs are private
        """
        self.df.dropna(subset=['Time', 'Source', 'Destination', 'Protocol', 'Length'], inplace=True)
        self.df['Time'] = pd.to_numeric(self.df['Time'], errors='coerce')
        self.df['Length'] = pd.to_numeric(self.df['Length'], errors='coerce')
        self.df.dropna(subset=['Time', 'Length'], inplace=True)
        self.df['Time'] -= self.df['Time'].min()
        self.df['Time'] /= self.df['Time'].max()
        self.df['is_local_src'] = self.df['Source'].apply(self._is_private_ip).astype(int)
        self.df['is_local_dst'] = self.df['Destination'].apply(self._is_private_ip).astype(int)

    def _is_private_ip(self, ip: str) -> bool:
        """
        Check whether an IP address is private.

        Parameters:
            ip (str): IP address as a string.

        Returns:
            bool: True if IP is private, False otherwise.
        """
        try:
            return ipaddress.ip_address(ip).is_private
        except ValueError:
            return False

    def _extract_unique_protocols(self) -> List[str]:
        """
        Extract and sort unique protocols from the dataset.

        Returns:
            List[str]: Sorted list of unique protocol names.
        """
        return sorted(self.df['Protocol'].unique())

    def vectorize_packet(self, row: pd.Series) -> List[float]:
        """
        Convert a single packet (row) into a numerical vector.

        Parameters:
            row (pd.Series): A single packet's data.

        Returns:
            List[float]: Vectorized representation of the packet.
        """
        vec: List[float] = [
            row['Time'],
            row['Length'],
            row['is_local_src'],
            row['is_local_dst']
        ]
        for proto in self.protocols:
            vec.append(1.0 if row['Protocol'] == proto else 0.0)
        return vec

    def get_feature_matrix(self) -> np.ndarray:
        """
        Build the complete feature matrix for all packets in the dataset.

        Returns:
            np.ndarray: 2D array where each row is a vectorized packet.
        """
        return np.array([self.vectorize_packet(row) for _, row in self.df.iterrows()])

    def apply_pca(self, n_components: float = 0.95) -> np.ndarray:
        """
        Apply PCA to the feature matrix to reduce its dimensionality.

        Parameters:
            n_components (float): Percentage of variance to preserve.

        Returns:
            np.ndarray: Transformed feature matrix after PCA.
        """
        X = self.get_feature_matrix()
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        self._pca = PCA(n_components=n_components)
        X_pca = self._pca.fit_transform(X_scaled)
        return X_pca

    def get_pca_model(self) -> Optional[PCA]:
        """
        Retrieve the fitted PCA model.

        Returns:
            Optional[PCA]: The fitted PCA model, or None if PCA has not been applied.
        """
        return self._pca

    def get_scaler(self) -> Optional[StandardScaler]:
        """
        Retrieve the fitted StandardScaler.

        Returns:
            Optional[StandardScaler]: The fitted scaler, or None if PCA has not been applied.
        """
        return self._scaler
