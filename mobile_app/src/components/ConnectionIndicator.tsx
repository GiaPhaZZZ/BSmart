/**
 * BSmart Connection Indicator Component
 * Shows BLE connection status: Connected / Connecting / Disconnected
 * See: docs/requirement.md §8
 */

import React from 'react';
import { StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { BleConnectionState } from '../types';

interface ConnectionIndicatorProps {
  state: BleConnectionState;
  onConnect: () => void;
  onDisconnect: () => void;
}

const STATE_CONFIG: Record<
  BleConnectionState,
  { color: string; label: string; dotColor: string }
> = {
  [BleConnectionState.CONNECTED]: {
    color: '#388E3C',
    label: 'Connected',
    dotColor: '#66BB6A',
  },
  [BleConnectionState.CONNECTING]: {
    color: '#F57C00',
    label: 'Connecting...',
    dotColor: '#FFB300',
  },
  [BleConnectionState.DISCONNECTED]: {
    color: '#C62828',
    label: 'Disconnected',
    dotColor: '#EF5350',
  },
};

export function ConnectionIndicator({
  state,
  onConnect,
  onDisconnect,
}: ConnectionIndicatorProps) {
  const cfg = STATE_CONFIG[state];
  const isConnected = state === BleConnectionState.CONNECTED;
  const isConnecting = state === BleConnectionState.CONNECTING;

  return (
    <View
      style={[styles.container, { borderColor: cfg.color + '40' }]}
      accessible={true}
      accessibilityRole="summary"
      accessibilityLabel={`Trạng thái Bluetooth: ${cfg.label}`}
    >
      <View style={styles.left}>
        <View style={[styles.dot, { backgroundColor: cfg.dotColor }]} />
        <View>
          <Text style={styles.bleLabel}>Bluetooth</Text>
          <Text style={[styles.stateText, { color: cfg.color }]}>
            {cfg.label}
          </Text>
        </View>
      </View>
      {isConnecting ? (
        <View
          accessible={true}
          accessibilityRole="alert"
          accessibilityLabel="Đang tìm và kết nối kính"
        >
          <Text style={styles.connectingText}>Đang quét...</Text>
        </View>
      ) : (
        <TouchableOpacity
          style={[
            styles.button,
            isConnected ? styles.buttonDisconnect : styles.buttonConnect,
          ]}
          onPress={isConnected ? onDisconnect : onConnect}
          accessible={true}
          accessibilityRole="button"
          accessibilityLabel={
            isConnected
              ? 'Ngắt kết nối Bluetooth'
              : 'Kết nối kính Bluetooth'
          }
          accessibilityHint={
            isConnected
              ? 'Nhấn hai lần để ngắt kết nối với kính'
              : 'Nhấn hai lần để bắt đầu quét và kết nối với kính AI'
          }
        >
          <Text style={styles.buttonText}>
            {isConnected ? 'Ngắt' : 'Kết nối'}
          </Text>
        </TouchableOpacity>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: '#0D1117',
    borderRadius: 10,
    borderWidth: 1,
    marginBottom: 12,
  },
  left: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  dot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginRight: 8,
  },
  bleLabel: {
    color: '#8B949E',
    fontSize: 11,
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  stateText: {
    fontSize: 14,
    fontWeight: '600',
    marginTop: 1,
  },
  button: {
    paddingHorizontal: 14,
    paddingVertical: 7,
    borderRadius: 8,
  },
  buttonConnect: {
    backgroundColor: '#1565C0',
  },
  buttonDisconnect: {
    backgroundColor: '#C62828',
  },
  buttonText: {
    color: '#FFFFFF',
    fontSize: 13,
    fontWeight: '600',
  },
  connectingText: {
    color: '#FFB300',
    fontSize: 13,
    fontWeight: '600',
    fontStyle: 'italic',
  },
});
