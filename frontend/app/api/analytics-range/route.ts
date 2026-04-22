import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  try {
    const token = request.headers.get('Authorization')?.replace('Bearer ', '');

    if (!token) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const startDate = request.nextUrl.searchParams.get('startDate') || '';
    const endDate = request.nextUrl.searchParams.get('endDate') || '';
    const analysisType = request.nextUrl.searchParams.get('analysisType') || 'daily';

    const response = await fetch(`http://127.0.0.1:5000/api/analytics/team-range?startDate=${startDate}&endDate=${endDate}&analysisType=${analysisType}`, {
      headers: {
        'Authorization': `Bearer ${token}`,
      },
    });

    const raw = await response.text();
    let data: unknown;
    try {
      data = raw ? JSON.parse(raw) : {};
    } catch {
      data = {
        error: 'Backend returned non-JSON response',
        details: raw.slice(0, 300),
      };
    }

    if (!response.ok) {
      return NextResponse.json(data, { status: response.status });
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error('Team analytics range API error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
