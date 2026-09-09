/**
 * BSmart Debug Log Component
 * Scrollable log of recent messages for developer debugging.
 * See: docs/requirement.md §8
 */

import React, { useEffect, useRef } from 'react';
import {
  FlatList,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { LogEntry } from '../types';

interface DebugLogProps {
  logs: LogEntry[];
}

const LOG_COLORS: Record<LogEntry['type'], string> = {
  info: '#8B949E',
  warn: '#FFB300',
  error: '#EF5350',
};

function LogRow({ item }: { item: LogEntry }) {
  return (
    <View style={styles.row}>
      <Text style={styles.timestamp}>{item.timestamp}</Text>
      <Text style={[styles.message, { color: LOG_COLORS[item.type] }]}>
        {item.message}
      </Text>
    </View>
  );
}

export function DebugLog({ logs }: DebugLogProps) {
  const flatListRef = useRef<FlatList>(null);

  useEffect(() => {
    if (logs.length > 0) {
      flatListRef.current?.scrollToIndex({ index: 0, animated: true });
    }
  }, [logs]);

  return (
    <View style={styles.container}>
      <Text style={styles.header}>Log</Text>
      <FlatList
        ref={flatListRef}
        data={logs}
        keyExtractor={item => item.id}
        renderItem={({ item }) => <LogRow item={item} />}
        style={styles.list}
        showsVerticalScrollIndicator={false}
        onScrollToIndexFailed={() => {}}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0D1117',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#21262D',
    overflow: 'hidden',
  },
  header: {
    color: '#6E7681',
    fontSize: 11,
    fontWeight: '600',
    letterSpacing: 1.5,
    textTransform: 'uppercase',
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#21262D',
  },
  list: {
    flex: 1,
  },
  row: {
    flexDirection: 'row',
    paddingHorizontal: 14,
    paddingVertical: 5,
    borderBottomWidth: 1,
    borderBottomColor: '#161B22',
  },
  timestamp: {
    color: '#3D4450',
    fontSize: 10,
    fontFamily: 'monospace',
    marginRight: 8,
    minWidth: 72,
    paddingTop: 1,
  },
  message: {
    fontSize: 12,
    fontFamily: 'monospace',
    flex: 1,
    flexWrap: 'wrap',
  },
});
