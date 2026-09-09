/**
 * BSmart Main Screen
 * Primary UI for the BSmart AI Glasses controller app.
 * See: docs/requirement.md §8
 */

import React from 'react';
import {
  SafeAreaView,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { useAppStateMachine } from '../hooks/useAppStateMachine';
import { StatusDisplay } from '../components/StatusDisplay';
import { ConnectionIndicator } from '../components/ConnectionIndicator';
import { DebugLog } from '../components/DebugLog';
import { MockControls } from '../components/MockControls';

export function MainScreen() {
  const {
    appState,
    connectionState,
    logs,
    useMock,
    onMockButtonHold,
    onMockButtonRelease,
    onMockShortPress,
    onMockImage,
    onConnect,
    onDisconnect,
  } = useAppStateMachine();

  return (
    <SafeAreaView style={styles.root}>
      <StatusBar barStyle="light-content" />
      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.appName}>BSmart</Text>
          <Text style={styles.appSubtitle}>Kính AI hỗ trợ người khiếm thị</Text>
        </View>

        {/* Connection */}
        <ConnectionIndicator
          state={connectionState}
          onConnect={onConnect}
          onDisconnect={onDisconnect}
        />

        {/* Current State */}
        <StatusDisplay appState={appState} />

        {/* Mock BLE Controls (dev only) */}
        {useMock && (
          <MockControls
            onButtonHold={onMockButtonHold}
            onButtonRelease={onMockButtonRelease}
            onShortPress={onMockShortPress}
            onImage={onMockImage}
          />
        )}

        {/* Debug Log */}
        <View style={styles.logContainer}>
          <DebugLog logs={logs} />
        </View>
      </ScrollView>
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
    alignItems: 'center',
    marginBottom: 20,
    paddingVertical: 12,
  },
  appName: {
    color: '#4FC3F7',
    fontSize: 32,
    fontWeight: '800',
    letterSpacing: 3,
    textTransform: 'uppercase',
  },
  appSubtitle: {
    color: '#4D5566',
    fontSize: 12,
    marginTop: 4,
    letterSpacing: 0.5,
  },
  logContainer: {
    height: 320,
  },
});
