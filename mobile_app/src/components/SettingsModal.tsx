/**
 * BSmart Settings Modal
 * Provides toggles for:
 *   - Chế độ Lập trình viên (Dev HUD: DebugLog & MockControls)
 *   - Mock BLE vs Real BLE
 *   - Dịch vụ chạy ngầm (Foreground Service)
 */

import React from 'react';
import {
  Modal,
  StyleSheet,
  Switch,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';

interface SettingsModalProps {
  visible: boolean;
  onClose: () => void;
  isDevMode: boolean;
  onToggleDevMode: (val: boolean) => void;
  isMockBle: boolean;
  onToggleMockBle: (val: boolean) => void;
  isBackgroundServiceActive: boolean;
  onToggleBackgroundService: (val: boolean) => void;
}

export function SettingsModal({
  visible,
  onClose,
  isDevMode,
  onToggleDevMode,
  isMockBle,
  onToggleMockBle,
  isBackgroundServiceActive,
  onToggleBackgroundService,
}: SettingsModalProps) {
  return (
    <Modal
      animationType="slide"
      transparent={true}
      visible={visible}
      onRequestClose={onClose}
    >
      <View style={styles.overlay}>
        <View
          style={styles.modalContent}
          accessible={true}
          accessibilityRole="none"
          accessibilityLabel="Bảng cài đặt hệ thống"
        >
          <View style={styles.header}>
            <Text style={styles.title}>Cài đặt hệ thống</Text>
            <TouchableOpacity
              onPress={onClose}
              style={styles.closeButton}
              accessible={true}
              accessibilityRole="button"
              accessibilityLabel="Đóng cài đặt"
            >
              <Text style={styles.closeButtonText}>✕</Text>
            </TouchableOpacity>
          </View>

          {/* Option 1: Dev HUD Mode */}
          <View style={styles.settingItem}>
            <View style={styles.textContainer}>
              <Text style={styles.settingLabel}>Chế độ Lập trình viên</Text>
              <Text style={styles.settingSubtext}>
                Hiển thị bảng nhật ký log và các nút bấm giả lập
              </Text>
            </View>
            <Switch
              value={isDevMode}
              onValueChange={onToggleDevMode}
              trackColor={{ false: '#374151', true: '#0288D1' }}
              thumbColor={isDevMode ? '#4FC3F7' : '#9CA3AF'}
              accessible={true}
              accessibilityLabel="Công tắc chế độ lập trình viên"
            />
          </View>

          {/* Option 2: Mock BLE Toggle */}
          <View style={styles.settingItem}>
            <View style={styles.textContainer}>
              <Text style={styles.settingLabel}>Chế độ Giả lập BLE (Mock)</Text>
              <Text style={styles.settingSubtext}>
                {isMockBle
                  ? 'Đang dùng Mock BLE (Không cần kính thật)'
                  : 'Đang dùng Bluetooth thật (Quét kính ESP32)'}
              </Text>
            </View>
            <Switch
              value={isMockBle}
              onValueChange={onToggleMockBle}
              trackColor={{ false: '#374151', true: '#388E3C' }}
              thumbColor={isMockBle ? '#81C784' : '#9CA3AF'}
              accessible={true}
              accessibilityLabel="Công tắc chế độ giả lập Bluetooth"
            />
          </View>

          {/* Option 3: Background Service */}
          <View style={styles.settingItem}>
            <View style={styles.textContainer}>
              <Text style={styles.settingLabel}>Duy trì chạy ngầm (Foreground)</Text>
              <Text style={styles.settingSubtext}>
                Ngăn Android tắt app khi khóa màn hình (Doze Mode)
              </Text>
            </View>
            <Switch
              value={isBackgroundServiceActive}
              onValueChange={onToggleBackgroundService}
              trackColor={{ false: '#374151', true: '#F57C00' }}
              thumbColor={isBackgroundServiceActive ? '#FFB74D' : '#9CA3AF'}
              accessible={true}
              accessibilityLabel="Công tắc duy trì chạy ngầm"
            />
          </View>

          <TouchableOpacity
            style={styles.doneButton}
            onPress={onClose}
            accessible={true}
            accessibilityRole="button"
            accessibilityLabel="Lưu và Đóng cài đặt"
          >
            <Text style={styles.doneButtonText}>Xác nhận</Text>
          </TouchableOpacity>
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.75)',
    justifyContent: 'flex-end',
  },
  modalContent: {
    backgroundColor: '#161B22',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    padding: 24,
    borderTopWidth: 1,
    borderColor: '#30363D',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
  },
  title: {
    fontSize: 20,
    fontWeight: '700',
    color: '#F0F6FC',
  },
  closeButton: {
    padding: 8,
  },
  closeButtonText: {
    fontSize: 20,
    color: '#8B949E',
    fontWeight: 'bold',
  },
  settingItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 14,
    borderBottomWidth: 1,
    borderColor: '#21262D',
  },
  textContainer: {
    flex: 1,
    paddingRight: 16,
  },
  settingLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: '#C9D1D9',
    marginBottom: 4,
  },
  settingSubtext: {
    fontSize: 12,
    color: '#8B949E',
    lineHeight: 16,
  },
  doneButton: {
    marginTop: 24,
    backgroundColor: '#1F6FEB',
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
  },
  doneButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '700',
  },
});
