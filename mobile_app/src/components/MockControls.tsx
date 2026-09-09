/**
 * BSmart Mock Controls Component
 * Developer panel for simulating glasses hardware events.
 * Only visible when USE_MOCK_BLE is enabled.
 */

import React from 'react';
import { StyleSheet, Text, TouchableOpacity, View } from 'react-native';

interface MockControlsProps {
  onButtonHold: () => void;
  onButtonRelease: () => void;
  onShortPress: () => void;
  onImage: () => void;
}

interface MockButtonProps {
  label: string;
  onPress: () => void;
  color?: string;
  id: string;
}

function MockButton({ label, onPress, color = '#1565C0', id }: MockButtonProps) {
  return (
    <TouchableOpacity
      style={[styles.button, { backgroundColor: color }]}
      onPress={onPress}
      accessibilityLabel={id}
    >
      <Text style={styles.buttonText}>{label}</Text>
    </TouchableOpacity>
  );
}

export function MockControls({
  onButtonHold,
  onButtonRelease,
  onShortPress,
  onImage,
}: MockControlsProps) {
  return (
    <View style={styles.container}>
      <Text style={styles.header}>🔧 Mock BLE Controls</Text>
      <View style={styles.row}>
        <MockButton
          id="mock-btn-hold"
          label="Hold Record"
          onPress={onButtonHold}
          color="#1B5E20"
        />
        <MockButton
          id="mock-btn-release"
          label="Release"
          onPress={onButtonRelease}
          color="#1565C0"
        />
        <MockButton
          id="mock-btn-short"
          label="Short Press"
          onPress={onShortPress}
          color="#B71C1C"
        />
        <MockButton
          id="mock-btn-image"
          label="Send Image"
          onPress={onImage}
          color="#4A148C"
        />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#0D1117',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#30363D',
    padding: 12,
    marginBottom: 12,
  },
  header: {
    color: '#6E7681',
    fontSize: 11,
    fontWeight: '600',
    letterSpacing: 1,
    textTransform: 'uppercase',
    marginBottom: 10,
  },
  row: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  button: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
  },
  buttonText: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '600',
  },
});
