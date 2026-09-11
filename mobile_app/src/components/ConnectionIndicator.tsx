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
  deviceName?: string | null;
  onConnect: () => void;
  onDisconnect: () => void;
}

const STATE_CONFIG: Record<
  BleConnectionState,
  { color: string; label: string; dotColor: string }
> = {
  [BleConnectionState.CONNECTED]: {
    color: '#388E3C',
    label: 'Đã kết nối kính',
    dotColor: '#66BB6A',
  },
  [BleConnectionState.CONNECTING]: {
    color: '#F57C00',
    label: 'Đang kết nối...',
    dotColor: '#FFB300',
  },
  [BleConnectionState.DISCONNECTED]: {
    color: '#8B949E',
    label: 'Chưa kết nối kính',
    dotColor: '#EF5350',
  },
  [BleConnectionState.BLUETOOTH_OFF]: {
    color: '#E53935',
    label: 'Bluetooth đang tắt',
    dotColor: '#757575',
  },
};

export function ConnectionIndicator({
  state,
  deviceName,
  onConnect,
  onDisconnect,
}: ConnectionIndicatorProps) {
  const cfg = STATE_CONFIG[state] || STATE_CONFIG[BleConnectionState.DISCONNECTED];
  const isConnected = state === BleConnectionState.CONNECTED;
  const isConnecting = state === BleConnectionState.CONNECTING;
  const isBleOff = state === BleConnectionState.BLUETOOTH_OFF;

  const currentLabel = isConnected
    ? `Đã kết nối thành công '${deviceName || 'BSmart Glasses'}'`
    : cfg.label;

  const getButtonAction = () => {
    if (isConnected) return onDisconnect;
    return onConnect;
  };

  const getButtonText = () => {
    if (isConnected) return 'Ngắt';
    if (isBleOff) return 'Bật Bluetooth';
    return 'Kết nối';
  };

  const getButtonStyle = () => {
    if (isConnected) return styles.buttonDisconnect;
    if (isBleOff) return styles.buttonEnable;
    return styles.buttonConnect;
  };

  return (
    <View
      style={[styles.container, { borderColor: cfg.color + '40' }]}
      accessible={true}
      accessibilityRole="summary"
      accessibilityLabel={`Trạng thái Bluetooth: ${currentLabel}`}
    >
      <View style={styles.left}>
        <View style={[styles.dot, { backgroundColor: cfg.dotColor }]} />
        <View style={styles.textContainer}>
          <Text style={styles.bleLabel}>Bluetooth</Text>
          <Text style={[styles.stateText, { color: cfg.color }]} numberOfLines={2}>
            {currentLabel}
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
          style={[styles.button, getButtonStyle()]}
          onPress={getButtonAction()}
          accessible={true}
          accessibilityRole="button"
          accessibilityLabel={
            isConnected
              ? 'Ngắt kết nối Bluetooth'
              : isBleOff
              ? 'Bật Bluetooth điện thoại'
              : 'Kết nối kính Bluetooth'
          }
          accessibilityHint={
            isConnected
              ? 'Nhấn hai lần để ngắt kết nối với kính'
              : isBleOff
              ? 'Nhấn hai lần để yêu cầu bật Bluetooth trên điện thoại'
              : 'Nhấn hai lần để bắt đầu quét và kết nối với kính AI'
          }
        >
          <Text style={styles.buttonText}>
            {getButtonText()}
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
    flex: 1,
    marginRight: 8,
  },
  textContainer: {
    flexShrink: 1,
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
  buttonEnable: {
    backgroundColor: '#F57C00',
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
