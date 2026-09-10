/**
 * BSmart Main Screen
 * Blind-First UI for the BSmart AI Glasses controller app.
 * Features:
 *   - Default Blind Mode ("Màn hình tiếp xúc rộng") with full TalkBack support
 *   - Push-to-talk hold/release and tap-to-cancel gestures with haptic vibration & beeps
 *   - Discreet Settings button to toggle Developer HUD (DebugLog & MockControls)
 *   - Real BLE by default with runtime Android permissions and background keep-alive
 */

import React, { useState } from 'react';
import {
  SafeAreaView,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { useAppStateMachine } from '../hooks/useAppStateMachine';
import { StatusDisplay } from '../components/StatusDisplay';
import { ConnectionIndicator } from '../components/ConnectionIndicator';
import { BlindTouchArea } from '../components/BlindTouchArea';
import { SettingsModal } from '../components/SettingsModal';
import { DebugLog } from '../components/DebugLog';
import { MockControls } from '../components/MockControls';
import {
  startBackgroundService,
  stopBackgroundService,
} from '../services/background/ForegroundService';

export function MainScreen() {
  const {
    appState,
    connectionState,
    logs,
    isMockBle,
    toggleMockBle,
    onButtonHold,
    onButtonRelease,
    onShortPress,
    onMockButtonHold,
    onMockButtonRelease,
    onMockShortPress,
    onMockImage,
    onConnect,
    onDisconnect,
  } = useAppStateMachine();

  // Settings & Modes
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isDevMode, setIsDevMode] = useState(false); // Default: Blind Mode (Dev HUD hidden)
  const [isBackgroundActive, setIsBackgroundActive] = useState(false);

  const handleToggleBackground = async (val: boolean) => {
    setIsBackgroundActive(val);
    if (val) {
      await startBackgroundService();
    } else {
      await stopBackgroundService();
    }
  };

  return (
    <SafeAreaView style={styles.root}>
      <StatusBar barStyle="light-content" />
      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* Header with Title & Settings Button */}
        <View
          style={styles.header}
          accessible={true}
          accessibilityRole="header"
          accessibilityLabel="BSmart Kính AI hỗ trợ người khiếm thị"
        >
          <View style={styles.headerTitles}>
            <Text style={styles.appName}>BSmart</Text>
            <Text style={styles.appSubtitle}>Kính AI hỗ trợ người khiếm thị</Text>
          </View>

          <TouchableOpacity
            style={styles.settingsButton}
            onPress={() => setIsSettingsOpen(true)}
            accessible={true}
            accessibilityRole="button"
            accessibilityLabel="Mở bảng cài đặt ứng dụng và chế độ lập trình viên"
            accessibilityHint="Nhấn hai lần để mở các tùy chọn cài đặt Bluetooth và nhà phát triển"
          >
            <Text style={styles.settingsIcon}>⚙️</Text>
          </TouchableOpacity>
        </View>

        {/* Bluetooth Connection Status */}
        <ConnectionIndicator
          state={connectionState}
          onConnect={onConnect}
          onDisconnect={onDisconnect}
        />

        {/* Current State Indicator */}
        <StatusDisplay appState={appState} />

        {/* Blind-First Touch Area (Always active and prominent) */}
        <BlindTouchArea
          appState={appState}
          onHold={onButtonHold}
          onRelease={onButtonRelease}
          onShortPress={onShortPress}
        />

        {/* Developer HUD Section (Hidden by default, enabled via Settings) */}
        {isDevMode && (
          <View style={styles.devContainer}>
            <View style={styles.devHeader}>
              <Text style={styles.devTitle}>🛠️ CHẾ ĐỘ LẬP TRÌNH VIÊN</Text>
              <Text style={styles.devSub}>
                {isMockBle ? 'Đang dùng Mock BLE' : 'Đang dùng Real BLE'}
              </Text>
            </View>

            {/* Mock BLE Controls (only when Mock BLE is enabled) */}
            {isMockBle && (
              <MockControls
                onButtonHold={onMockButtonHold}
                onButtonRelease={onMockButtonRelease}
                onShortPress={onMockShortPress}
                onImage={onMockImage}
              />
            )}

            {/* Live Debug Log */}
            <View style={styles.logContainer}>
              <DebugLog logs={logs} />
            </View>
          </View>
        )}
      </ScrollView>

      {/* Settings Modal */}
      <SettingsModal
        visible={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        isDevMode={isDevMode}
        onToggleDevMode={setIsDevMode}
        isMockBle={isMockBle}
        onToggleMockBle={toggleMockBle}
        isBackgroundServiceActive={isBackgroundActive}
        onToggleBackgroundService={handleToggleBackground}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: '#010409',
  },
  scroll: {
    flex: 1,
  },
  scrollContent: {
    padding: 16,
    paddingBottom: 32,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 16,
    paddingVertical: 8,
  },
  headerTitles: {
    flex: 1,
  },
  appName: {
    color: '#4FC3F7',
    fontSize: 28,
    fontWeight: '800',
    letterSpacing: 2,
    textTransform: 'uppercase',
  },
  appSubtitle: {
    color: '#8B949E',
    fontSize: 13,
    marginTop: 2,
    letterSpacing: 0.5,
  },
  settingsButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: '#161B22',
    borderWidth: 1,
    borderColor: '#30363D',
    alignItems: 'center',
    justifyContent: 'center',
  },
  settingsIcon: {
    fontSize: 20,
  },
  devContainer: {
    marginTop: 16,
    paddingTop: 16,
    borderTopWidth: 1,
    borderColor: '#21262D',
  },
  devHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  devTitle: {
    color: '#FFB74D',
    fontSize: 13,
    fontWeight: '700',
    letterSpacing: 1,
  },
  devSub: {
    color: '#8B949E',
    fontSize: 11,
  },
  logContainer: {
    height: 280,
    marginTop: 10,
  },
});
