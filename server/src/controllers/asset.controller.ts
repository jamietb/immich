import { Body, Controller, Delete, Get, HttpCode, HttpStatus, Param, Post, Put, Query } from '@nestjs/common';
import { ApiOperation, ApiTags } from '@nestjs/swagger';
import { EndpointLifecycle } from 'src/decorators';
import { AssetResponseDto } from 'src/dtos/asset-response.dto';
import {
  AssetBulkDeleteDto,
  AssetBulkUpdateDto,
  AssetJobsDto,
  AssetStatsDto,
  AssetStatsResponseDto,
  DeviceIdDto,
  RandomAssetsDto,
  UpdateAssetDto,
} from 'src/dtos/asset.dto';
import { AuthDto } from 'src/dtos/auth.dto';
import { Permission, RouteKey } from 'src/enum';
import { Auth, Authenticated } from 'src/middleware/auth.guard';
import { AssetService } from 'src/services/asset.service';
import { UUIDParamDto } from 'src/validation';

@ApiTags('Assets')
@Controller(RouteKey.Asset)
export class AssetController {
@Get(':id/similar')
@Authenticated({ permission: Permission.AssetRead })
@ApiOperation({ summary: 'Find visually similar assets' })
async getSimilarAssets(
  @Auth() auth: AuthDto,
  @Param() { id }: UUIDParamDto,
  @Query('types') types?: string,
  @Query('limit') limit?: string,
  @Query('offset') offset?: string,
  @Query('stacksOnly') stacksOnly?: string,
  @Query('minDate') minDate?: string,
  @Query('maxDate') maxDate?: string,
  @Query('albumId') albumId?: string,
  @Query('cameraMake') cameraMake?: string,
  @Query('cameraModel') cameraModel?: string,
) {
  const parsedTypes =
    types
      ? types.split(',').map((t) => t.trim().toUpperCase()).filter((t) => t === 'IMAGE' || t === 'VIDEO')
      : undefined;

  const parsedLimit = Math.max(1, Math.min(500, Number(limit) || 100));
  const parsedOffset = Math.max(0, Number(offset) || 0);
  const parsedStacks = (stacksOnly ?? 'true').toLowerCase() !== 'false';

  const items = await this.service.findSimilarAssets({
    userId: auth.user.id,
    assetId: id,
    types: parsedTypes as any,
    limit: parsedLimit,
    offset: parsedOffset,
    stacksOnly: parsedStacks,
    minDate: minDate || null,
    maxDate: maxDate || null,
    albumId: albumId || null,
    cameraMake: cameraMake || null,
    cameraModel: cameraModel || null,
  });

  return { items };
}

  constructor(private service: AssetService) {}

  @Get('random')
  @Authenticated({ permission: Permission.AssetRead })
  @EndpointLifecycle({ deprecatedAt: 'v1.116.0' })
  getRandom(@Auth() auth: AuthDto, @Query() dto: RandomAssetsDto): Promise<AssetResponseDto[]> {
    return this.service.getRandom(auth, dto.count ?? 1);
  }

  /**
   * Get all asset of a device that are in the database, ID only.
   */
  @Get('/device/:deviceId')
  @ApiOperation({
    summary: 'getAllUserAssetsByDeviceId',
    description: 'Get all asset of a device that are in the database, ID only.',
  })
  @Authenticated()
  getAllUserAssetsByDeviceId(@Auth() auth: AuthDto, @Param() { deviceId }: DeviceIdDto) {
    return this.service.getUserAssetsByDeviceId(auth, deviceId);
  }

  @Get('statistics')
  @Authenticated({ permission: Permission.AssetStatistics })
  getAssetStatistics(@Auth() auth: AuthDto, @Query() dto: AssetStatsDto): Promise<AssetStatsResponseDto> {
    return this.service.getStatistics(auth, dto);
  }

  @Post('jobs')
  @Authenticated()
  @HttpCode(HttpStatus.NO_CONTENT)
  runAssetJobs(@Auth() auth: AuthDto, @Body() dto: AssetJobsDto): Promise<void> {
    return this.service.run(auth, dto);
  }

  @Put()
  @Authenticated({ permission: Permission.AssetUpdate })
  @HttpCode(HttpStatus.NO_CONTENT)
  updateAssets(@Auth() auth: AuthDto, @Body() dto: AssetBulkUpdateDto): Promise<void> {
    return this.service.updateAll(auth, dto);
  }

  @Delete()
  @Authenticated({ permission: Permission.AssetDelete })
  @HttpCode(HttpStatus.NO_CONTENT)
  deleteAssets(@Auth() auth: AuthDto, @Body() dto: AssetBulkDeleteDto): Promise<void> {
    return this.service.deleteAll(auth, dto);
  }

  @Get(':id')
  @Authenticated({ permission: Permission.AssetRead, sharedLink: true })
  getAssetInfo(@Auth() auth: AuthDto, @Param() { id }: UUIDParamDto): Promise<AssetResponseDto> {
    return this.service.get(auth, id) as Promise<AssetResponseDto>;
  }

  @Put(':id')
  @Authenticated({ permission: Permission.AssetUpdate })
  updateAsset(
    @Auth() auth: AuthDto,
    @Param() { id }: UUIDParamDto,
    @Body() dto: UpdateAssetDto,
  ): Promise<AssetResponseDto> {
    return this.service.update(auth, id, dto);
  }
}
